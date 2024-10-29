-- Full file path: /moneyverse/database/configurations_table.sql

CREATE TABLE configurations (
    config_key VARCHAR(50) PRIMARY KEY,
    config_value VARCHAR(255),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
