use std::borrow::Borrow;
use std::fs;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use arrow::array::{Array, StringArray};
use clap::Parser;
use fastbloom::{AtomicBloomFilter, BloomFilter};
use hashbrown::HashSet;
use parquet::{
    arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    errors::ParquetError,
    file::reader::{FileReader, SerializedFileReader},
    record::{Row, RowAccessor},
};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use thiserror::Error;

enum Container<T> {
    Hash(HashSet<T>),
    Bloom(BloomFilter),
    ParHash(Mutex<HashSet<T>>),
    ParBloom(AtomicBloomFilter),
}

#[allow(clippy::multiple_bound_locations)]
impl<T: Hash + Eq> Container<T> {
    fn insert(&mut self, value: T) {
        match self {
            Self::Hash(set) => set.insert(value),
            Self::Bloom(filter) => filter.insert(&value),
            Self::ParHash(m_set) => {
                let lock = m_set.get_mut().unwrap();
                lock.insert(value)
            }
            Self::ParBloom(m_filter) => m_filter.insert(&value),
        };
    }

    fn contains<Q: ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self {
            Self::Hash(set) => set.contains(value),
            Self::Bloom(filter) => filter.contains(value),
            Self::ParHash(m_set) => {
                let lock = m_set.lock().unwrap();
                lock.contains(value)
            }
            Self::ParBloom(m_filter) => m_filter.contains(value),
        }
    }

    fn new(bloom: bool, parallel: bool) -> Self
    where
        T: Hash + Eq,
    {
        match (bloom, parallel) {
            (true, true) => Self::ParBloom(
                AtomicBloomFilter::with_false_pos(0.001).expected_items(2_000_000_000),
            ),
            (true, false) => {
                Self::Bloom(BloomFilter::with_false_pos(0.001).expected_items(2_000_000_000))
            }
            (false, true) => Self::ParHash(Mutex::new(HashSet::new())),
            (false, false) => Self::Hash(HashSet::new()),
        }
    }

    fn try_union(&mut self, bloom: AtomicBloomFilter) {
        if let Container::ParBloom(filter) = self {
            filter.union(&bloom)
        }
    }
}

impl<'a, T: Hash + Eq + Clone + 'a> Container<T> {
    fn extend<Q: IntoIterator<Item = &'a T> + Iterator<Item = &'a T>>(&mut self, value: Q)
    where
        T: Hash + Eq + Clone,
    {
        match self {
            Self::Hash(set) => set.extend(value.cloned()),
            Self::Bloom(filter) => filter.insert_all(value),
            Self::ParHash(m_set) => {
                let lock = m_set.get_mut().unwrap();
                lock.extend(value.cloned());
            }
            Self::ParBloom(m_filter) => m_filter.insert_all(value),
        }
    }
}

impl<T: Hash + Eq> From<HashSet<T>> for Container<T> {
    fn from(value: HashSet<T>) -> Self {
        Container::Hash(value)
    }
}

impl<T: Hash + Eq> From<Mutex<HashSet<T>>> for Container<T> {
    fn from(value: Mutex<HashSet<T>>) -> Self {
        Container::ParHash(value)
    }
}

impl<T: Hash + Eq> From<BloomFilter> for Container<T> {
    fn from(value: BloomFilter) -> Self {
        Container::Bloom(value)
    }
}

impl<T: Hash + Eq> From<AtomicBloomFilter> for Container<T> {
    fn from(value: AtomicBloomFilter) -> Self {
        Container::ParBloom(value)
    }
}
#[derive(Debug, Error)]
enum Error {
    #[error("Error opening file")]
    Reading(#[from] std::io::Error),
    #[error("Error writing to file")]
    Writing(#[from] serde_json::Error),
    #[error("Error with parquet file")]
    Parquet(#[from] ParquetError),
    #[error("Error setting up parallelization")]
    Threading(#[from] rayon::ThreadPoolBuildError),
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// left, the baseline (this should should have less content)
    /// automatic detection for a folder of parquet files or a single parquet file
    #[arg(short, long)]
    left: String,

    /// right, the comparison (this should have more content)
    /// automatic detection for a folder of parquet files or a single parquet file
    #[arg(short, long)]
    right: String,

    /// file to write the diff to (will output in a list, json format)
    #[arg(short, long)]
    output: String,

    /// use a bloomfilter instead of a hashset
    #[arg(short, long)]
    bloomfilter: bool,

    /// use parallel iterators
    #[arg(short, long)]
    parallel: bool,

    /// number of threads to use
    /// warning if not supplied it will use them all, depending on how much data you are parsing
    /// this will probably eat all of your RAM quickly...
    #[arg(short, long)]
    num_threads: Option<u32>,
}

fn find_parquet_files(dir: &String) -> Result<Vec<PathBuf>, Error> {
    let dir_iter = std::fs::read_dir(Path::new(dir))?;
    let mut files = vec![];

    for file in dir_iter.flatten() {
        if file
            .file_name()
            .to_ascii_lowercase()
            .to_string_lossy()
            .ends_with("parquet")
        {
            files.push(file.path());
        }
    }

    Ok(files)
}

fn parquet_reader(file: &PathBuf) -> Result<Vec<Result<Row, ParquetError>>, Error> {
    let pq_file = std::fs::File::open(file)?;
    let reader = SerializedFileReader::new(pq_file)?;

    Ok(reader
        .get_row_iter(None)?
        .collect::<Vec<Result<Row, ParquetError>>>())
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    if let Some(threads) = args.num_threads
        && args.parallel
    {
        println!("Using {threads} threads");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build_global()?;
    }

    let mut file_names = Container::new(args.bloomfilter, args.parallel);

    let outfile = std::fs::File::create_new(&args.output)?;
    let buf = std::io::BufWriter::new(outfile);

    let left_metadata = fs::metadata(&args.left)?;
    let left_file_type = left_metadata.file_type();

    // left file/dir is our base we open it/all of them and hold it in the set/filter
    if left_file_type.is_file() {
        let file_path = PathBuf::from(&args.left);
        let mut rows = parquet_reader(&file_path)?;
        println!("opened file: {} to check against", args.left);
        if args.parallel {
            let all_file_names = rows
                .into_par_iter()
                .fold(
                    || AtomicBloomFilter::with_false_pos(0.001).expected_items(2_000_000_000),
                    |local_names, record| {
                        // TODO handle this a little safer
                        if let Ok(file_name) = record.unwrap().get_string(0).cloned() {
                            local_names.insert(&file_name);
                        }

                        local_names
                    },
                )
                .reduce(
                    || AtomicBloomFilter::with_false_pos(0.001).expected_items(2_000_000_000),
                    |global, locals| {
                        global.union(&locals);

                        global
                    },
                );

            file_names.try_union(all_file_names);
        } else {
            while let Some(record) = rows.pop() {
                if let Ok(file_name) = record.as_ref().unwrap().get_string(0) {
                    file_names.insert(file_name.clone());
                }
            }
        }
    } else {
        // dir crawl and open all parquet files if this is a directory
        let files = find_parquet_files(&args.left)?;
        if args.parallel {
            //todo return result from closure to handle errors better?
            let all_file_names = files
                .into_par_iter()
                .map(|file_handle| {
                    let file = fs::File::open(&file_handle).unwrap();
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
                    let reader = builder.with_batch_size(8192).build().unwrap();

                    //let mut rows = parquet_reader(&file);
                    println!("opened file: {} to check against", file_handle.display());
                    reader
                        .par_bridge()
                        .into_par_iter()
                        .fold(
                            || {
                                AtomicBloomFilter::with_false_pos(0.001)
                                    .expected_items(2_000_000_000)
                            },
                            |local_names, record| {
                                let record_batch = record.unwrap();

                                let record_arrays = record_batch
                                    .column(0)
                                    .as_any()
                                    .downcast_ref::<StringArray>()
                                    .unwrap();

                                record_arrays.iter().par_bridge().into_par_iter().for_each(
                                    |record| {
                                        if let Some(file_name) = record {
                                            local_names.insert(file_name);
                                        }
                                    },
                                );

                                local_names
                            },
                        )
                        .reduce(
                            || {
                                AtomicBloomFilter::with_false_pos(0.001)
                                    .expected_items(2_000_000_000)
                            },
                            |global, locals| {
                                global.union(&locals);

                                global
                            },
                        )
                })
                .reduce(
                    || AtomicBloomFilter::with_false_pos(0.001).expected_items(2_000_000_000),
                    |global, locals| {
                        global.union(&locals);

                        global
                    },
                );

            file_names.try_union(all_file_names);
        } else {
            for file in files {
                let mut rows = parquet_reader(&file)?;
                println!("opened file: {} to check against", file.to_string_lossy());

                while let Some(record) = rows.pop() {
                    if let Ok(file_name) = record.as_ref().unwrap().get_string(0) {
                        file_names.insert(file_name.clone());
                    }
                }
            }
        }
    }

    println!("Done opening base files");

    let right_metadata = fs::metadata(&args.right)?;
    let right_file_type = right_metadata.file_type();

    // then check each file in the right dir and keep track of missing files
    let mut missing_files: HashSet<String> = HashSet::new();

    if right_file_type.is_file() {
        let file_path = PathBuf::from(&args.right);
        let mut rows = parquet_reader(&file_path)?;
        println!("Opened file {} to compare", args.right);
        if args.parallel {
            let temp = rows
                .into_par_iter()
                .map(|record| match record.as_ref().unwrap().get_string(0) {
                    Ok(file_name) => file_name.to_owned(),
                    _ => String::new(),
                })
                .filter(|x| !x.is_empty())
                .collect::<Vec<String>>();

            file_names.extend(temp.iter());
        } else {
            while let Some(record) = rows.pop() {
                if let Ok(file_name) = record.as_ref().unwrap().get_string(0) {
                    file_names.insert(file_name.clone());
                }
            }
        }
    } else {
        // right is a directory so we must directory crawl
        let files = find_parquet_files(&args.right)?;

        if args.parallel {
            // TODO return result from closure to handle errors better?
            let missing = files
                .into_par_iter()
                .map(|file_handle| {
                    let file = fs::File::open(&file_handle).unwrap();
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
                    let reader = builder.with_batch_size(20000).build().unwrap();

                    println!("opened file: {} to check against", file_handle.display());
                    reader
                        .par_bridge()
                        .into_par_iter()
                        .map(|record| {
                            let record_batch = record.unwrap();
                            let record_arrays = record_batch
                                .column(0)
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap();

                            record_arrays
                                .iter()
                                .par_bridge()
                                .into_par_iter()
                                .filter(|x| x.is_some())
                                .map(|z| z.unwrap())
                                .filter(|y| !file_names.contains(*y))
                                .fold(HashSet::new, |mut missing_files, x| {
                                    missing_files.insert(x.to_string());

                                    missing_files
                                })
                                .reduce(HashSet::new, |mut global, locals| {
                                    global.extend(locals);

                                    global
                                })
                        })
                        .fold(HashSet::new, |mut local_missing, missing_file| {
                            local_missing.extend(missing_file);
                            local_missing
                        })
                        .reduce(HashSet::new, |mut global, locals| {
                            global.extend(locals);

                            global
                        })
                })
                .reduce(HashSet::new, |mut global, locals| {
                    global.extend(locals);

                    global
                });

            missing_files = missing;
        } else {
            for file in files {
                let mut rows = parquet_reader(&file)?;
                println!("opened file: {} to compare", file.to_string_lossy());

                while let Some(record) = rows.pop() {
                    if let Ok(file_name) = record.as_ref().unwrap().get_string(0)
                        && !file_names.contains(file_name)
                    {
                        missing_files.insert(file_name.to_string());
                    }
                }
            }
        }
    }

    println!("Done comparing files\n######################");

    serde_json::to_writer_pretty(buf, &missing_files)?;
    println!("successfully wrote missing files to: {}", args.output);

    Ok(())
}
