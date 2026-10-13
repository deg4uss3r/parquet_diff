# parquet_diff

A fast [ Apache `parquet`](https://parquet.apache.org) file diff tool written in Rust.

## Running

The most common (and right now best way) to call this program is: 

`/target/release/parquet_diff -l /baseline/path/ -r /comparison/path --bloomfilter --parallel --num-threads 10 --output /path/output.csv`

This through testing is the most performant way to run the program. Through testing this compares about 60GBs of files (20GB left, 40GB right) in 30 minutes producing a 8GB 100_000_000 line CSV file. 

The options below are provided to help fine tune the program running to your system. 

## Options

```
❯ parquet_diff --help
Usage: parquet_diff [OPTIONS] --left <LEFT> --right <RIGHT> --output <OUTPUT>

Options:
  -l, --left <LEFT>                left, the baseline (this should should have less content) automatic detection for a folder of parquet files or a single parquet file
  -r, --right <RIGHT>              right, the comparison (this should have more content) automatic detection for a folder of parquet files or a single parquet file
  -o, --output <OUTPUT>            file to write the diff to (output will be file path,size(in bytes) csv format)
  -b, --bloomfilter                use a bloomfilter instead of a hashset
  -p, --parallel                   use parallel iterators
  -n, --num-threads <NUM_THREADS>  number of threads to use warning if not supplied it will use them all, depending on how much data you are parsing this will probably eat all of your CPU
  -b, --blocks <BLOCKS>            size of bytes to read at a single time from parquet files default: 8192 (8KB) [default: 8192]
  -h, --help                       Print help
  -V, --version                    Print version
```

## Building

The program is designed to be fast as possible during runtime so the compiler options are tuned for runtime performance not compiler performance. It takes about 30s to build on an M2 Pro machine, your experience might be a lot worse depending on your build environment.

## Open Issues

Currently, the program is focused on left and right directory options and parallel processing using the bloomfilter flag. Other options have bugs currently due to the focus being runtime optimization, the goal is to clean those up and have them run through the same path or eliminate those options all together.