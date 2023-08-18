DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    carat FLOAT,
    cut TEXT,
    color TEXT,
    clarity TEXT,
    depth FLOAT,
    `table` FLOAT,
    price BIGINT,
    x FLOAT,
    y FLOAT,
    z FLOAT
) DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    `playerID` VARCHAR(255),
    `yearID` BIGINT,
    stint BIGINT,
    `teamID` VARCHAR(7),
    `lgID` VARCHAR(7),
    `G` BIGINT,
    `AB` BIGINT,
    `R` BIGINT,
    `H` BIGINT,
    `X2B` BIGINT,
    `X3B` BIGINT,
    `HR` BIGINT,
    `RBI` BIGINT,
    `SB` BIGINT,
    `CS` BIGINT,
    `BB` BIGINT,
    `SO` BIGINT,
    `IBB` BIGINT,
    `HBP` BIGINT,
    `SH` BIGINT,
    `SF` BIGINT,
    `GIDP` BIGINT
) DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    `playerID` VARCHAR(255),
    `awardID` VARCHAR(255),
    `yearID` BIGINT,
    `lgID` VARCHAR(7),
    tie VARCHAR(7),
    notes VARCHAR(255)
) DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    id INTEGER,
    bool_col BOOLEAN,
    tinyint_col TINYINT,
    smallint_col SMALLINT,
    int_col INTEGER,
    bigint_col BIGINT,
    float_col FLOAT,
    double_col DOUBLE,
    date_string_col TEXT,
    string_col TEXT,
    timestamp_col DATETIME,
    year INTEGER,
    month INTEGER
) DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS json_t;

CREATE TABLE IF NOT EXISTS json_t (js JSON);

INSERT INTO json_t VALUES
    ('{"a": [1,2,3,4], "b": 1}'),
    ('{"a":null,"b":2}'),
    ('{"a":"foo", "c":null}'),
    ('null'),
    ('[42,47,55]'),
    ('[]');

DROP TABLE IF EXISTS win;

CREATE TABLE win (g TEXT, x BIGINT, y BIGINT);
INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);

DROP TABLE IF EXISTS datatypes;

CREATE TABLE datatypes (
    id INT UNSIGNED,
    tinyint_c TINYINT,
    smallint_c SMALLINT,
    mediumint_c MEDIUMINT,
    int_c INT,
    bigint_c BIGINT,
    float_c FLOAT,
    double_c DOUBLE,
    decimal_c DECIMAL(18,6),
    decimal5_c DECIMAL(18,5),
    numeric_c NUMERIC,
    date_c DATE,
    time_c TIME,
    time6_c TIME(6),
    datetime_c DATETIME,
    datetime6_c DATETIME(6),
    timestamp_c TIMESTAMP,
    timestamp6_c TIMESTAMP(6),
    char32_c CHAR(32),
    varchar42_c VARCHAR(42),
    longtext_c LONGTEXT,
    mediumtext_c MEDIUMTEXT,
    tinytext_c TINYTEXT,
    text_c TEXT,
    text4_c TEXT(4),
    blob_c BLOB,
    enum_sml_c ENUM('small', 'medium', 'large'),
    set_abcd_c SET('a', 'b', 'c', 'd'),
    negative_int_c INT,
    negative_float_c FLOAT,
    bool_c BOOL
);
