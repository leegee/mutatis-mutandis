-- compare liberty across slices
SELECT slice_start, neighbour, rank
FROM neighbourhoods
WHERE query = 'liberty'
  AND rank <= 10
ORDER BY slice_start, rank;
