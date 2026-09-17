"""
Log Message Preprocessor for BertGCN Pipeline.
Section 4.2 of Alsalmi et al. (CMC 2025/2026):

The paper explicitly uses a PARSING-FREE approach where raw log messages are fed
directly to BERT (WordPiece tokenizer handles subword tokenization).
Cleaning steps applied (as described in Sec 4.2):
  - Strip leading/trailing whitespace
  - Remove common log metadata prefixes (timestamps, hostnames, PIDs, severity labels)
  - Remove hexadecimal memory addresses (0x...)
  - Collapse multiple whitespace tokens into a single space
  - Lowercase (BERT base uncased expects this)

This intentionally does NOT apply regex-based template extraction (e.g., Drain/AEL),
preserving the full semantic context for BERT.
"""
import re
from typing import List


class LogPreprocessor:
    """
    Parsing-free log message cleaner as described in Alsalmi et al. Sec 4.2.
    Strips timestamps, severity labels, host names, PIDs, and hex addresses
    while retaining the core semantic content of each log message.
    """

    # Patterns to strip from log lines before passing to BERT
    _PATTERNS = [
        # ISO-8601 / syslog timestamps: 2023-11-01T13:45:23.123 or 081109 203615
        re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"),
        re.compile(r"\d{6}\s\d{6}"),          # BGL compact timestamp: 081109 203615
        # Syslog / HDFS style leading metadata:
        # e.g.  "Nov 01 13:45:23 hostname sshd[1234]:"  OR  "081109 203615 0 FATAL ..."
        re.compile(r"^(?:[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+\S+\s+"),
        # BGL line prefix: "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ..."
        re.compile(r"^-\s+\d+\s+[\d.]+\s+\S+\s+"),
        # Hex addresses: 0x7f3a4b2c1d
        re.compile(r"0x[0-9a-fA-F]+"),
        # Numeric-only tokens (pure numbers add noise; keep if mixed alphanumeric)
        # We leave numbers that are part of words (e.g. blk_-1234 → keep -1234 context)
        # PID-like [1234] or (1234)
        re.compile(r"\[\d+\]"),
        re.compile(r"\(\d+\)"),
        # Severity labels at the start of messages (after timestamp strip)
        re.compile(r"^(?:FATAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE|NOTICE|CRITICAL)\s+", re.IGNORECASE),
        # Log level in angle/square brackets: [INFO], [ERROR]
        re.compile(r"\[(?:FATAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE|NOTICE|CRITICAL)\]", re.IGNORECASE),
        # Multiple whitespace
        re.compile(r"\s{2,}"),
    ]

    def clean_log_message(self, raw_message: str) -> str:
        """
        Applies the parsing-free cleaning pipeline to a raw log line.

        Args:
            raw_message: A single raw log line string.

        Returns:
            Cleaned, lowercased log message ready for BERT tokenization.
        """
        text = raw_message.strip()

        for pattern in self._PATTERNS:
            if pattern.pattern == r"\s{2,}":
                text = pattern.sub(" ", text)
            else:
                text = pattern.sub(" ", text)

        text = text.strip().lower()
        # Guard: return a placeholder if message becomes empty after cleaning
        return text if text else "[empty]"

    def clean_sequence(self, messages: List[str]) -> List[str]:
        """
        Cleans an entire sequence of log messages.

        Args:
            messages: List of raw log line strings.

        Returns:
            List of cleaned, lowercased strings.
        """
        return [self.clean_log_message(m) for m in messages]
