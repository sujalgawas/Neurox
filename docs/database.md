# Database and data models

The application does not use a standalone relational database engine. Instead, data persistence and state management are divided between in-memory rolling feature histories, an append-only JSONL signal log configured via `combined_signal_log_path`, and broker-managed account and order records accessed through the Alpaca paper-trading API.

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"| -->
```mermaid
erDiagram
    BAR_RECORD ||--o{ MARKET_OBSERVATION : composes
    MARKET_OBSERVATION ||--|| JEPA_PREDICTION : evaluates
    MARKET_OBSERVATION ||--|| JEV_PREDICTION : evaluates
    JEPA_PREDICTION ||--|| COMBINED_SIGNAL : inputs
    JEV_PREDICTION ||--|| COMBINED_SIGNAL : inputs
    COMBINED_SIGNAL ||--o| SIGNAL_LOG : appends
    COMBINED_SIGNAL ||--o| ORDER : triggers
    ACCOUNT ||--o{ POSITION : holds
    ACCOUNT ||--o{ ORDER : executes

    BAR_RECORD {
        string timestamp
        float open
        float high
        float low
        float close
        int volume
        float vwap
    }

    MARKET_OBSERVATION {
        string symbol
        float latest_price
        list raw_records
        list jepa_features
    }

    JEPA_PREDICTION {
        int direction
        string direction_label
        float direction_confidence
        float latest_price
    }

    JEV_PREDICTION {
        int direction
        string direction_label
        float confidence
        float latest_price
    }

    COMBINED_SIGNAL {
        string symbol
        string action
        float combined_score
        float size
        string rule
        float normalized_jepa
        float normalized_jev
    }

    SIGNAL_LOG {
        string timestamp
        string symbol
        string action
        float combined_score
        string rule
    }

    ORDER {
        string order_id PK
        string symbol
        string side
        float qty
        string status
        string time_in_force
    }

    POSITION {
        string symbol PK
        float qty
        float avg_entry_price
    }

    ACCOUNT {
        float cash
        float equity
        float buying_power
        boolean trading_blocked
    }
```

## Data stores and state management

### In-memory rolling history

The `CombinedPipeline` class maintains an in-memory dictionary mapping each uppercase equity symbol to rolling score magnitude lists:

- `histories[symbol]["jepa"]` — a list of prior absolute JEPA direction scores, capped to `COMBINED_PERCENTILE_WINDOW` entries (default 500).
- `histories[symbol]["jev"]` — a list of prior absolute JEV confidence scores, capped to `COMBINED_PERCENTILE_WINDOW` entries (default 500).
- These lists reset when the server restarts; they are not written to disk.

### Signal log file

When signal logging is enabled, predictions are appended as newline-delimited JSON (JSONL) records to the path defined by `combined_signal_log_path` (default: `data/combined/signals.jsonl`). Each line records the timestamp, symbol, action, scores, rule, and whether orders were submitted.

### Alpaca broker state

All persistent order, position, and account records reside on Alpaca servers and are queried on demand via `alpaca-py`:

- **Account:** queried before placing buy orders to verify that `trading_blocked` is false and `buying_power` is sufficient.
- **Positions:** queried before order placement to determine whether to open, close, or reverse long and short holdings.
- **Orders:** active orders are inspected by `_available_order_quantity()` in [`app/Trader/trading.py`](../app/Trader/trading.py) to prevent submitting duplicate sell or cover orders for shares already committed to pending transactions.

