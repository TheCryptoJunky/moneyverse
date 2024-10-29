-- Full file path: moneyverse/database/configurations_table.sql

-- Configuration Table for System Settings
CREATE TABLE IF NOT EXISTS configurations (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,             -- Key name of the configuration
    config_value TEXT NOT NULL,                          -- Value associated with the configuration
    description TEXT,                                    -- Optional description of the configuration
    data_type VARCHAR(20) DEFAULT 'string',              -- Data type: e.g., string, integer, float, boolean
    is_encrypted BOOLEAN DEFAULT FALSE,                  -- Flag to indicate if the config_value is encrypted
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,    -- Timestamp of the last update
    updated_by VARCHAR(50)                               -- Identifier of the updater (username or process)
);

-- Example Insertions for Configurations Aligned with Recent Updates
INSERT INTO configurations (config_key, config_value, description, data_type, is_encrypted, updated_by)
VALUES 
    -- Reinforcement Learning Decision Threshold
    ('rl_decision_threshold', '0.7', 'Threshold for RL-based action confidence', 'float', FALSE, 'system'),

    -- Reinvestment Interval (seconds)
    ('reinvestment_interval', '3600', 'Interval for periodic profit reinvestment in seconds', 'integer', FALSE, 'system'),

    -- Encryption Key for Sensitive Data
    ('encryption_key', '<encryption_key_here>', 'Encryption key for sensitive wallet data', 'string', TRUE, 'admin'),

    -- Flask GUI Session Timeout
    ('flask_gui_timeout', '30', 'Session timeout for Flask GUI in minutes', 'integer', FALSE, 'system'),

    -- AI Model Selection
    ('ai_model', 'PPO', 'AI model to use for NAV optimization', 'string', FALSE, 'system'),

    -- Aggregator Selection Criteria
    ('aggregator_selection', 'cost-effective', 'Strategy for choosing asset aggregators', 'string', FALSE, 'system'),

    -- Wallet Rebalance Threshold (USD)
    ('rebalance_threshold', '5000', 'Threshold for triggering wallet rebalancing', 'float', FALSE, 'admin')
ON CONFLICT (config_key) DO NOTHING;  -- Avoid duplicate entries

-- Communications Table for Notification Settings
CREATE TABLE IF NOT EXISTS communications (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,              -- e.g., "email", "sms", "telegram", "discord"
    encrypted_value BYTEA NOT NULL          -- Encrypted contact info for each type
);
