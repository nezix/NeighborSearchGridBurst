# Fast neighbor search using a grid in Unity, Job System + Burst 
---------------

## This is a fast, parallel (C# job system) and vectorized (Burst compiled) code for Unity for k-neighbor search.
For some data, it can be faster than using a KDtree (can be 25 times faster than KNN from https://github.com/ArthurBrussee/KNN.

#### Best case scenario: uniformly distributed points in space
#### Worst case scenario: nearly all points close to each others and some points far from the cluster

- **For k-neighbor search, if maxNeighborPerQuery is not enough, the search does not return all points close to the query point. It can skip points that are closer to the one returned**

Parameters to adjust:
- Grid resolution, adjust this parameter according to your data, density per cell shouldn't be too large for performance, not too low for memory consumption
- For k-neighbor search, you have to specify the maximum number of neighbors (maxNeighborPerQuery).

Example cases:
- Points ranging from -100 to 100 in x, y and/or z: for a grid resolution of 5, the grid size will be 200 / 5 => (40 x 40 x 40)
- Points ranging from 0 to 0.05, for a grid resolution of 0.00156, the grid size will be 0.05 / 0.00156 => (32 x 32 x 32)

#### Tested on Unity 2022.3.61

## Examples

1) How to get all closest point of a point cloud ?
2) How to get k neighbor points of a point cloud ?

```C#
  using BurstGridSearch;
  ...

  Vector3[] positions = new Vector3[N];
  Vector3[] queries = new Vector3[K];
  //Fill the arrays...
  ...
  
  float gridReso = 5.0f;//Adjust this parameter according to your data, density per cell shouldn't be too large for performance, not too low for memory consumption
  //If gridReso < 0, the resolution will be adjusted so that the grid is 32 x 32 x 32 (this can be changed by doing: new GridSearchBurst(-1.0f, 64);)
  GridSearchBurst gsb = new GridSearchBurst(gridReso);
  
  gsb.InitializeGrid(positions).Complete();
  //For each query point, find the closest point
  int[] results = gsb.SearchClosestPoint(queries);
 
   //For each query point, find the closest point that is not exactly the same 
  int[] results2 = gsb.SearchClosestPoint(queries, checkSelf: true, epsilon: 0.001f);
  
  //For each query point, find neighbor points in a radius of 2 (only 50 points are searched)
  int[] results3 = gsb.SearchWithin(queries, 2.0f, 50);
  //Result array contains 50 x len(queries) indices in positions array
  
  //Print the position of the closest point to the first query point
  if(results[0] != -1)
    Debug.Log(positions[results[0]);
  
  
  gsb.Dispose();//Free up the native arrays !
  
```

## Benchmark: Grid vs KDtree (KNN implementation from: https://github.com/ArthurBrussee/KNN ) (Intel i7 4790k)

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

When initializing, all points are assigned to a cell in an uniform grid. The resolution of the grid should depend on your data, make sure to __adjust this parameter for maximum performance and memory consumption__. You can also target a size grid by doing ```new GridSearchBurst(-1.0f, 64);``` to have a grid of 64x64x64.

For each point, the computed 3D grid cell is flatten into an unique hash and a couple ```(hash, index)``` is stored in the ```hashIndex``` array. This array is sorted by hash (thanks to the sorting code from https://coffeebraingames.wordpress.com/2020/06/07/a-multithreaded-sorting-attempt) so that each points in the same cell are successive in the array.


For neighbor searching, the grid cell of the query point is computed, first we search this cell for neighbors using the sorted array that acts as a sort of neighbor list. If k neighbors were find we stop, otherwise we check neighbor cells for points that are close enough.

## Installation
Add the git url to the package manager: https://docs.unity3d.com/Manual/upm-ui-giturl.html

[<img src="https://docs.unity3d.com/Packages/Installation/manual/images/PackageManagerUI-GitURLPackageButton.png">](https://docs.unity3d.com/Manual/upm-ui-giturl.html)

## Contribute

Pull requests are more than welcome!

## License


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
