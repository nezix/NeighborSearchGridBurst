# Fast neighbor search using a grid in Unity, Job System + Burst 

#### Tested on Unity 2019.4.21

## Example

```C#
  Vector3[] positions = new Vector3[N];
  Vector3[] queries = new Vector3[K];
  //Fill the arrays...
  ...
  
  float gridReso = 5.0f;//Adjust this parameter according to your data, density per cell shouldn't be too large for performance, not too low for memory consumption
  //If gridReso < 0, the resolution will be adjusted so that the grid is 32 x 32 x 32 (this can be changed by doing: new GridSearchBurst(-1.0f, 64);)
  GridSearchBurst gsb = new GridSearchBurst(gridReso);
  
  gsb.initGrid(positions);
  //For each query point, find the closest point
  int[] results = gsb.searchClosestPoint(queries);
 
   //For each query point, find the closest point that is not exactly the same 
  int[] results2 = gsb.searchClosestPoint(queries, checkSelf: true, epsilon: 0.001f);
  
  //For each query point, find neighbor points in a radius of 2 (only 50 points are searched)
  int[] results3 = gsb.searchWithin(queries, 2.0f, 50);
  //Result array contains 50 x len(queries) indices in positions array
  
  //Print the position of the closest point to the first query point
  if(results[0] != -1)
    Debug.Log(positions[results[0]);
  
  
```

## Grid vs KDtree (KNN implementation from: [https://github.com/ArthurBrussee/KNN](here)) (Intel i7 4790k)
##### i7 4790k

- on 100k random positions between (-100, 100) and 10k queries: 

|               | Grid   | KNN    |
|---------------|--------|------- |
| Setup         | 4ms    | 8.7ms  |
| Closest Point | 0.18ms | 7.7ms  |
| Total         | 4.18ms | 16.4ms |

- on 100k random positions and 100k queries:

|               | Grid  | KNN   |
|---------------|-------|-------|
| Setup         | 4.5ms | 9ms   |
| Closest Point | 2.3ms | 180ms |
| Total         | 6.8ms | 189ms |

- on 100k random positions and 100k queries with a radius search of 2 and 50 neighbor max:

|                 | Grid   | KNN   |
|-----------------|--------|-------|
| Setup           | 4.6ms  | 8.7ms   |
| Neighbor search | 23.2ms | 398ms   |
| Total           | 27.8ms | 406.7ms |

## Details

When initializing, all points are distributed in a uniform grid. The resolution of the grid should depend on your data, make sure to __adjust this parameter for maximum performance and memory consumption__. You can also target a size grid by doing ```new GridSearchBurst(-1.0f, 64);``` to have a grid of 64x64x64.


## Contribute

Pull requests are more than welcome!

## License


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
