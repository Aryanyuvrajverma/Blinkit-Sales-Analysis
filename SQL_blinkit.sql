 create database Blinkit_db;
 use blinkit_db;
 CREATE TABLE sales (
    Item_Identifier VARCHAR(20),
    Item_Weight FLOAT,
    Item_Fat_Content VARCHAR(20),
    Item_Visibility FLOAT,
    Item_Type VARCHAR(50),
    Item_MRP FLOAT,
    Outlet_Identifier VARCHAR(20),
    Outlet_Establishment_Year INT,
    Outlet_Size VARCHAR(20),
    Outlet_Location_Type VARCHAR(20),
    Outlet_Type VARCHAR(30),
    Item_Outlet_Sales FLOAT
);
SELECT COUNT(*) FROM sales;
SHOW WARNINGS;

TRUNCATE TABLE sales;

LOAD DATA LOCAL INFILE 'D:/Project2/blinkit_clean.csv'
INTO TABLE sales
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

SET GLOBAL local_infile = 1;

SELECT 
    COUNT(*) AS total_rows,
    COUNT(Item_Weight) AS weight_count,
    COUNT(Outlet_Size) AS size_count
FROM sales;

SELECT SUM(Item_Outlet_Sales) AS Total_Sales FROM sales;

SELECT Outlet_Type, SUM(Item_Outlet_Sales) AS Sales
FROM sales
GROUP BY Outlet_Type
ORDER BY Sales DESC;

SELECT Outlet_Type, AVG(Item_Outlet_Sales) AS Avg_Sales
FROM sales
GROUP BY Outlet_Type
ORDER BY Avg_Sales DESC;

SELECT Outlet_Location_Type, SUM(Item_Outlet_Sales)
FROM sales
GROUP BY Outlet_Location_Type;

SELECT Item_Type, SUM(Item_Outlet_Sales) AS Sales
FROM sales
GROUP BY Item_Type
ORDER BY Sales DESC;

SELECT 
    Outlet_Type,
    SUM(Item_Outlet_Sales) AS Total_Sales,
    RANK() OVER (ORDER BY SUM(Item_Outlet_Sales) DESC) AS Rank_Position
FROM sales
GROUP BY Outlet_Type;

SELECT 
    Outlet_Type,
    SUM(Item_Outlet_Sales) AS Sales,
    ROUND(SUM(Item_Outlet_Sales) * 100 / 
(SELECT SUM(Item_Outlet_Sales) FROM sales), 2) AS Contribution_Percentage
FROM sales
GROUP BY Outlet_Type;

SELECT 
    CASE 
        WHEN Item_Visibility < 0.05 THEN 'Low Visibility'
        ELSE 'High Visibility'
    END AS Visibility_Level,
    AVG(Item_Outlet_Sales) AS Avg_Sales
FROM sales
GROUP BY Visibility_Level;

SELECT Item_Fat_Content, AVG(Item_Outlet_Sales)
FROM sales
GROUP BY Item_Fat_Content;

SELECT 
    CASE 
        WHEN Outlet_Establishment_Year < 2000 THEN 'Old'
        ELSE 'New'
    END AS Outlet_Age,
    AVG(Item_Outlet_Sales) AS Avg_Sales
FROM sales
GROUP BY Outlet_Age;

SELECT Item_Identifier, SUM(Item_Outlet_Sales) AS Sales
FROM sales
GROUP BY Item_Identifier
ORDER BY Sales DESC
LIMIT 10;

SELECT Outlet_Type, AVG(Item_Outlet_Sales) AS Avg_Sales
FROM sales
GROUP BY Outlet_Type
ORDER BY Avg_Sales ASC;