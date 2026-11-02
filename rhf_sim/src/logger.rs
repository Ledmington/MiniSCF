//! Global logger of `rhf_sim`: records go to the terminal (through `simple_logger`) until
//! `--log-file` is opened, and only to that file afterwards. Errors are also printed to
//! the terminal, so that a failing run never exits silently.

use log::{Level, Log, Metadata, Record};
use simple_logger::SimpleLogger;
use std::fs::File;
use std::io::{LineWriter, Write};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

struct RhfLogger {
    terminal: SimpleLogger,
    /// Line-buffered, so that the file is complete even if the process exits abruptly.
    file: Mutex<Option<LineWriter<File>>>,
    /// Log file lines are timestamped with the time elapsed since this instant.
    start: Instant,
}

static LOGGER: OnceLock<RhfLogger> = OnceLock::new();

/// Installs the global logger, writing to the terminal. Call it once, at the very start
/// of `main`: log file timestamps count from this call.
pub fn init() {
    let rhf_logger = LOGGER.get_or_init(|| RhfLogger {
        terminal: SimpleLogger::new().env().without_timestamps(),
        file: Mutex::new(None),
        start: Instant::now(),
    });
    log::set_logger(rhf_logger).expect("the logger must be initialized only once");
    log::set_max_level(rhf_logger.terminal.max_level());
}

/// Sends every following record to `file` instead of the terminal.
pub fn redirect_to_file(file: File) {
    let rhf_logger = LOGGER.get().expect("logger::init must be called first");
    *lock(&rhf_logger.file) = Some(LineWriter::new(file));
}

fn lock(
    file: &Mutex<Option<LineWriter<File>>>,
) -> std::sync::MutexGuard<'_, Option<LineWriter<File>>> {
    // A panic while holding the lock leaves the writer usable, so ignore poisoning.
    file.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Formats a record as a log file line: time elapsed since startup (`HH:MM:SS.mmm`),
/// level, logger name and message.
fn format_line(elapsed: Duration, record: &Record) -> String {
    let millis = elapsed.as_millis();
    let (hours, minutes, seconds) = (millis / 3_600_000, millis / 60_000 % 60, millis / 1000 % 60);
    format!(
        "{hours:02}:{minutes:02}:{seconds:02}.{:03} {:<5} [{}] {}\n",
        millis % 1000,
        record.level(),
        record.target(),
        record.args()
    )
}

impl Log for RhfLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        self.terminal.enabled(metadata)
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let mut file = lock(&self.file);
        match file.as_mut() {
            Some(writer) => {
                let line = format_line(self.start.elapsed(), record);
                // There is nowhere left to report a failure to write the log itself.
                let _ = writer.write_all(line.as_bytes());
                if record.level() == Level::Error {
                    self.terminal.log(record);
                }
            }
            None => self.terminal.log(record),
        }
    }

    fn flush(&self) {
        if let Some(writer) = lock(&self.file).as_mut() {
            let _ = writer.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warning() -> Record<'static> {
        Record::builder()
            .level(Level::Warn)
            .target("scf_sim::sim")
            .args(format_args!("not converged"))
            .build()
    }

    #[test]
    fn log_file_lines_have_elapsed_time_level_logger_and_message() {
        let line = format_line(
            Duration::from_micros(3_723_045_678),
            &Record::builder()
                .level(Level::Warn)
                .target("scf_sim::sim")
                .args(format_args!("not converged after {} iterations", 100))
                .build(),
        );
        assert_eq!(
            "01:02:03.045 WARN  [scf_sim::sim] not converged after 100 iterations\n",
            line
        );
    }

    #[test]
    fn elapsed_time_is_zero_padded_and_hours_do_not_wrap() {
        assert!(format_line(Duration::ZERO, &warning()).starts_with("00:00:00.000 "));
        assert!(
            format_line(Duration::from_millis(59_999), &warning()).starts_with("00:00:59.999 ")
        );
        assert!(
            format_line(Duration::from_secs(100 * 3600), &warning()).starts_with("100:00:00.000 ")
        );
    }
}
