use std::borrow::Borrow;
use std::fs;
use std::hash::Hash;
use std::io::Write;
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
use rayon::iter::{
    IntoParallelIterator, ParallelBridge, ParallelIterator,
};
use thiserror::Error;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

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
    /// this will probably eat all of your CPU
    #[arg(short, long)]
    num_threads: Option<u32>,

    /// size of bytes to read at a single time from parquet files
    /// default: 8192 (8KB)
    #[arg(short, long, default_value_t = 8192)]
    blocks: usize,
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

    let file_names = Mutex::new(Container::new(args.bloomfilter, args.parallel));

    let outfile = std::fs::File::create_new(&args.output)?;

    let left_metadata = fs::metadata(&args.left)?;
    let left_file_type = left_metadata.file_type();

    // left file/dir is our base we open it/all of them and hold it in the set/filter
    if left_file_type.is_file() {
        let file_path = PathBuf::from(&args.left);
        let mut rows = parquet_reader(&file_path)?;
        println!("opened file: {} to check against", args.left);
        if args.parallel {
            rows.into_par_iter().for_each(|record| {
                // TODO handle this a little safer
                if let Ok(file_name) = record.unwrap().get_string(0).cloned() {
                    file_names.lock().unwrap().insert(file_name);
                }
            });
        } else {
            while let Some(record) = rows.pop() {
                if let Ok(file_name) = record.as_ref().unwrap().get_string(0) {
                    file_names.lock().unwrap().insert(file_name.clone());
                }
            }
        }
    } else {
        // dir crawl and open all parquet files if this is a directory
        let files = find_parquet_files(&args.left)?;
        if args.parallel {
            //todo return result from closure to handle errors better?
            files.into_iter().for_each(|file_handle| {
                let file = fs::File::open(&file_handle).unwrap();
                let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
                let reader = builder.with_batch_size(args.blocks).build().unwrap();

                //let mut rows = parquet_reader(&file);
                println!("opened file: {} to check against", file_handle.display());
                reader.into_iter().par_bridge().for_each(|record| {
                    let record_batch = record.unwrap();

                    let record_arrays = record_batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap();

                    record_arrays
                        .into_iter()
                        .par_bridge()
                        .for_each(|record| {
                            if let Some(file_name) = record {
                                file_names.lock().unwrap().insert(file_name.to_string());
                            }
                        });
                })
            });
        } else {
            for file in files {
                let mut rows = parquet_reader(&file)?;
                println!("opened file: {} to check against", file.to_string_lossy());

                while let Some(record) = rows.pop() {
                    if let Ok(file_name) = record.as_ref().unwrap().get_string(0) {
                        file_names.lock().unwrap().insert(file_name.clone());
                    }
                }
            }
        }
    }

    println!("Done opening base files");

    let right_metadata = fs::metadata(&args.right)?;
    let right_file_type = right_metadata.file_type();

    // then check each file in the right dir and keep track of missing files
    let missing_files: Mutex<HashSet<String>> = Mutex::new(HashSet::new());

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

            file_names.lock().unwrap().extend(temp.iter());
        } else {
            while let Some(record) = rows.pop() {
                if let Ok(file_name) = record.as_ref().unwrap().get_string(0) {
                    file_names.lock().unwrap().insert(file_name.clone());
                }
            }
        }
    } else {
        // right is a directory so we must directory crawl
        let files = find_parquet_files(&args.right)?;

        if args.parallel {
            // TODO return result from closure to handle errors better?
            files
                .into_iter()
                .for_each(|file_handle| {
                    let file = fs::File::open(&file_handle).unwrap();
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
                    let reader = builder.with_batch_size(args.blocks).build().unwrap();

                    println!("opened file: {} to check against", file_handle.display());
                    reader
                        .into_iter()
                        .par_bridge()
                        .for_each(|record| {
                            let record_batch = record.unwrap();
                            let record_arrays = record_batch
                                .column_by_name("Object_Name")
                                .unwrap()
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap();

                            let filesize_arrays = record_batch
                                .column_by_name("Size_Bytes")
                                .unwrap()
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap();

                            record_arrays
                                .into_iter()
                                .zip(filesize_arrays.into_iter())
                                .par_bridge()
                                .filter(|(file_name, file_size) | file_name.is_some() && !file_names.lock().unwrap().contains(file_name.unwrap()))
                                .for_each(|(missing_file, file_size)| {
                                    missing_files.lock().unwrap().insert(format!("{},{}", missing_file.unwrap(), file_size.unwrap()));
                                });
                        });
                });
        } else {
            for file in files {
                let mut rows = parquet_reader(&file)?;
                println!("opened file: {} to compare", file.to_string_lossy());

                while let Some(record) = rows.pop() {
                    if let Ok(file_name) = record.as_ref().unwrap().get_string(0)
                        && !file_names.lock().unwrap().contains(file_name)
                    {
                        missing_files.lock().unwrap().insert(file_name.to_string());
                    }
                }
            }
        }
    }

    println!("Done comparing files\n######################");

    // finally open the output file for writing
    // 8MB pages
    let mut buf = std::io::BufWriter::new( outfile);
    // pull out of the mutex to not be limited by disk I/O contention
    buf.write_all(b"file_name,file_size(bytes)\n")?;
    let contents = {
        let mut lock = missing_files.lock().unwrap();
        std::mem::take(&mut *lock) 
    };
    
    for line in contents.into_iter() {
        buf.write_all(line.as_bytes())?;
        buf.write_all(b"\n")?;
    }
    
    buf.flush()?;
    println!("successfully wrote missing files to: {}", args.output);

    Ok(())
}
