WITH slices AS (
    SELECT DISTINCT
        query,
        slice_start,
        slice_end
    FROM neighbourhoods
),
top_neighbours AS (
    SELECT
        query,
        slice_start,
        slice_end,
        neighbour
    FROM neighbourhoods
    WHERE rank = 1
),
dominance AS (
    SELECT
        query,
        neighbour,
        COUNT(*) AS times_top
    FROM top_neighbours
    GROUP BY query, neighbour
),
slice_counts AS (
    SELECT
        query,
        COUNT(*) AS total_slices
    FROM slices
    GROUP BY query
)
SELECT
    d.query,
    d.neighbour,
    d.times_top,
    s.total_slices
FROM dominance d
JOIN slice_counts s USING (query)
WHERE d.times_top = s.total_slices
ORDER BY d.query;
