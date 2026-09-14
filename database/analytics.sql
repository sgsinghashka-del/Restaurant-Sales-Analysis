-- ============================================================
-- RESTAURANT SALES ANALYSIS & INTELLIGENCE PLATFORM
-- BUSINESS ANALYTICS QUERIES
-- ============================================================


-- ============================================================
-- 1. OVERALL BUSINESS PERFORMANCE
-- ============================================================

SELECT
    COUNT(DISTINCT oi.order_id) AS total_orders,
    SUM(oi.quantity) AS total_items_sold,
    ROUND(SUM(oi.revenue), 2) AS total_revenue,
    ROUND(SUM(oi.ingredient_cost), 2) AS total_ingredient_cost,
    ROUND(SUM(oi.profit), 2) AS total_profit,
    ROUND(
        SUM(oi.profit) * 100.0 / NULLIF(SUM(oi.revenue), 0),
        2
    ) AS profit_margin_percentage
FROM order_items oi;


-- ============================================================
-- 2. TOP 10 SELLING DISHES
-- ============================================================

SELECT
    d.dish_id,
    d.dish_name,
    d.category,
    SUM(oi.quantity) AS total_quantity_sold,
    ROUND(SUM(oi.revenue), 2) AS total_revenue,
    ROUND(SUM(oi.profit), 2) AS total_profit
FROM order_items oi
JOIN dishes d
    ON oi.dish_id = d.dish_id
GROUP BY
    d.dish_id,
    d.dish_name,
    d.category
ORDER BY total_quantity_sold DESC
LIMIT 10;


-- ============================================================
-- 3. TOP 10 MOST PROFITABLE DISHES
-- ============================================================

SELECT
    d.dish_name,
    d.category,
    SUM(oi.quantity) AS quantity_sold,
    ROUND(SUM(oi.revenue), 2) AS revenue,
    ROUND(SUM(oi.profit), 2) AS profit,
    ROUND(
        SUM(oi.profit) * 100.0 /
        NULLIF(SUM(oi.revenue), 0),
        2
    ) AS profit_margin_percentage
FROM order_items oi
JOIN dishes d
    ON oi.dish_id = d.dish_id
GROUP BY
    d.dish_id,
    d.dish_name,
    d.category
ORDER BY profit DESC
LIMIT 10;


-- ============================================================
-- 4. HIGH SALES / LOW PROFIT DISHES
-- ============================================================

SELECT
    d.dish_name,
    d.category,
    SUM(oi.quantity) AS quantity_sold,
    ROUND(SUM(oi.revenue), 2) AS revenue,
    ROUND(SUM(oi.profit), 2) AS profit,
    ROUND(
        SUM(oi.profit) * 100.0 /
        NULLIF(SUM(oi.revenue), 0),
        2
    ) AS profit_margin_percentage
FROM order_items oi
JOIN dishes d
    ON oi.dish_id = d.dish_id
GROUP BY
    d.dish_id,
    d.dish_name,
    d.category
HAVING
    SUM(oi.quantity) >= 15000
    AND
    SUM(oi.profit) * 1.0 /
    NULLIF(SUM(oi.revenue), 0) < 0.45
ORDER BY quantity_sold DESC;


-- ============================================================
-- 5. CATEGORY-WISE PERFORMANCE
-- ============================================================

SELECT
    d.category,
    SUM(oi.quantity) AS quantity_sold,
    ROUND(SUM(oi.revenue), 2) AS revenue,
    ROUND(SUM(oi.profit), 2) AS profit,
    ROUND(
        SUM(oi.profit) * 100.0 /
        NULLIF(SUM(oi.revenue), 0),
        2
    ) AS profit_margin_percentage
FROM order_items oi
JOIN dishes d
    ON oi.dish_id = d.dish_id
GROUP BY d.category
ORDER BY revenue DESC;


-- ============================================================
-- 6. AREA-WISE PERFORMANCE
-- ============================================================

SELECT
    a.area_name,
    a.city,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(oi.revenue), 2) AS revenue,
    ROUND(SUM(oi.profit), 2) AS profit,
    ROUND(
        SUM(oi.profit) * 100.0 /
        NULLIF(SUM(oi.revenue), 0),
        2
    ) AS profit_margin_percentage
FROM order_items oi
JOIN orders o
    ON oi.order_id = o.order_id
JOIN areas a
    ON o.area_id = a.area_id
GROUP BY
    a.area_id,
    a.area_name,
    a.city
ORDER BY revenue DESC;


-- ============================================================
-- 7. MONTHLY SALES TREND
-- ============================================================

SELECT
    SUBSTR(o.order_date, 1, 7) AS month,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(oi.quantity) AS items_sold,
    ROUND(SUM(oi.revenue), 2) AS revenue,
    ROUND(SUM(oi.profit), 2) AS profit
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
GROUP BY month
ORDER BY month;


-- ============================================================
-- 8. PAYMENT METHOD ANALYSIS
-- ============================================================

SELECT
    o.payment_method,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(oi.revenue), 2) AS revenue,
    ROUND(AVG(o.order_value), 2) AS average_order_value
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
GROUP BY o.payment_method
ORDER BY revenue DESC;


-- ============================================================
-- 9. INVENTORY WASTAGE ANALYSIS
-- ============================================================

SELECT
    rm.material_name,
    rm.category,
    SUM(i.quantity_purchased) AS quantity_purchased,
    SUM(i.quantity_used) AS quantity_used,
    SUM(i.quantity_wasted) AS quantity_wasted,
    ROUND(
        SUM(i.quantity_wasted) * 100.0 /
        NULLIF(SUM(i.quantity_purchased), 0),
        2
    ) AS wastage_percentage
FROM inventory i
JOIN raw_materials rm
    ON i.material_id = rm.material_id
GROUP BY
    rm.material_id,
    rm.material_name,
    rm.category
ORDER BY wastage_percentage DESC;


-- ============================================================
-- 10. LOW STOCK MATERIALS
-- ============================================================

SELECT
    rm.material_id,
    rm.material_name,
    rm.category,
    rm.reorder_level,
    ROUND(
        SUM(i.quantity_purchased)
        - SUM(i.quantity_used)
        - SUM(i.quantity_wasted),
        2
    ) AS estimated_stock
FROM raw_materials rm
JOIN inventory i
    ON rm.material_id = i.material_id
GROUP BY
    rm.material_id,
    rm.material_name,
    rm.category,
    rm.reorder_level
HAVING estimated_stock < rm.reorder_level
ORDER BY estimated_stock ASC;


-- ============================================================
-- 11. SUPPLIER PERFORMANCE
-- ============================================================

SELECT
    s.supplier_name,
    s.supplier_city,
    s.rating,
    COUNT(DISTINCT rm.material_id) AS materials_supplied,
    ROUND(
        SUM(i.quantity_wasted) * 100.0 /
        NULLIF(SUM(i.quantity_purchased), 0),
        2
    ) AS wastage_percentage
FROM suppliers s
JOIN raw_materials rm
    ON s.supplier_id = rm.supplier_id
JOIN inventory i
    ON rm.material_id = i.material_id
GROUP BY
    s.supplier_id,
    s.supplier_name,
    s.supplier_city,
    s.rating
ORDER BY wastage_percentage ASC;


-- ============================================================
-- 12. TOP AREAS BY PROFIT
-- ============================================================

SELECT
    a.area_name,
    a.city,
    ROUND(SUM(oi.profit), 2) AS total_profit,
    ROUND(SUM(oi.revenue), 2) AS total_revenue
FROM order_items oi
JOIN orders o
    ON oi.order_id = o.order_id
JOIN areas a
    ON o.area_id = a.area_id
GROUP BY
    a.area_id,
    a.area_name,
    a.city
ORDER BY total_profit DESC;


-- ============================================================
-- END OF ANALYTICS
-- ============================================================