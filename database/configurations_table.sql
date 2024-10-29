-- Full file path: moneyverse/database/configurations_table.sql

CREATE TABLE configurations (
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

-- Reinforcement Learning Decision Threshold
INSERT INTO configurations (config_key, config_value, description, data_type, is_encrypted, updated_by)
VALUES 
('rl_decision_threshold', '0.7', 'Threshold for RL-based action confidence', 'float', FALSE, 'system');

-- Reinvestment Interval (seconds)
INSERT INTO configurations (config_key, config_value, description, data_type, is_encrypted, updated_by)
VALUES 
('reinvestment_interval', '3600', 'Interval for periodic profit reinvestment in seconds', 'integer', FALSE, 'system');

-- Encryption Key for Sensitive Data
INSERT INTO configurations (config_key, config_value, description, data_type, is_encrypted, updated_by)
VALUES 
('encryption_key', '<encryption_key_here>', 'Encryption key for sensitive wallet data', 'string', TRUE, 'admin');

-- Flask GUI Settings
INSERT INTO configurations (config_key, config_value, description, data_type, is_encrypted, updated_by)
VALUES 
('flask_gui_timeout', '30', 'Session timeout for Flask GUI in minutes', 'integer', FALSE, 'system');

-- AI Model Selection (e.g., PPO for reinforcement learning)
INSERT INTO configurations (config_key, config_value, description, data_type, is_encrypted, updated_by)
VALUES 
('ai_model', 'PPO', 'AI model to use for NAV optimization', 'string', FALSE, 'system');

-- Aggregator Selection Criteria
INSERT INTO configurations (config_key, config_value, description, data_type, is_encrypted, updated_by)
VALUES 
('aggregator_selection', 'cost-effective', 'Strategy for choosing asset aggregators', 'string', FALSE, 'system');

-- Wallet Rebalance Threshold (USD)
INSERT INTO configurations (config_key, config_value, description, data_type, is_encrypted, updated_by)
VALUES 
('rebalance_threshold', '5000', 'Threshold for triggering wallet rebalancing', 'float', FALSE, 'admin');
