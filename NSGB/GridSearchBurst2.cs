using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Exception = System.Exception;

namespace BurstGridSearch
{
    public class GridSearchBurst2 : IDisposable
    {
        private const int MAXGRIDSIZE = 256;
        private float _gridResolution = -1.0f;
        private int _targetGridSize;

        private NativeArray<float3> _positions;
        private NativeArray<float3> _sortedPositions;
        private NativeArray<int2> _hashIndex;
        private NativeList<int2> _cellStartEnd;
        private NativeArray<float3> _minMaxPositions;
        private NativeArray<int3> _gridDimensions;
        
        public GridSearchBurst2(float resolution, int targetGrid = 32)
        {
            if (resolution <= 0.0f && targetGrid > 0)
            {
                _targetGridSize = targetGrid;
                return;
            }
            if (resolution <= 0.0f && targetGrid <= 0)
            {
                throw new Exception("Wrong target grid size. Choose a resolution > 0 or a target grid > 0");
            }
            _gridResolution = resolution;
        }

        public JobHandle InitializeGrid(NativeArray<float3> positions)
        {
            Dispose();
            _positions = new NativeArray<float3>(positions.Length, Allocator.Persistent);
            _hashIndex = new (positions.Length, Allocator.Persistent);
            _sortedPositions = new (positions.Length, Allocator.Persistent);
            _cellStartEnd = new (Allocator.Persistent);
            positions.CopyTo(_positions);
            return InitializeGridInternal();
        }
        
        public JobHandle InitializeGrid(Vector3[] positions)
        {
            Dispose();
            _positions = new NativeArray<Vector3>(positions, Allocator.Persistent).Reinterpret<float3>();
            _hashIndex = new (positions.Length, Allocator.Persistent);
            _sortedPositions = new (positions.Length, Allocator.Persistent);
            _cellStartEnd = new (Allocator.Persistent);
            return InitializeGridInternal();
        }
        
        private JobHandle InitializeGridInternal()
        {
            if (_positions.Length == 0)
            {
                throw new Exception("Empty position buffer");
            }

            _cellStartEnd.Clear();
            _minMaxPositions = new (2, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            _gridDimensions = new (1, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            
            var job = new GridInializationJob()
            {
                Positions = _positions,
                GridResolution = _gridResolution,
                TargetGridSize = _targetGridSize,
                GridDimensions = _gridDimensions,
                CellStartEnd = _cellStartEnd,
                MinMaxPositions = _minMaxPositions,
            };
            var handle = job.Schedule();

            var assignHashJob = new AssignHashJob()
            {
                GridOrigin = _minMaxPositions,
                GridResolutionInv = 1.0f / _gridResolution,
                GridDimensions = _gridDimensions,
                Positions = _positions,
                HashIndex = _hashIndex,
                CellCount = _cellStartEnd.AsDeferredJobArray().Length,
            };
            handle = assignHashJob.Schedule(_positions.Length, 128, handle);
            
            NativeArray<SortEntry> entries = new NativeArray<SortEntry>(_positions.Length, Allocator.TempJob);
            
            var populateJob = new PopulateEntryJob()
            {
                HashIndex = _hashIndex,
                Entries = entries
            };
            
            handle = populateJob.Schedule(_positions.Length, 128, handle);
            
            // ------- Sort by hash
            handle = MultithreadedSort.Sort(entries, handle);
            
            var depopulateJob = new DePopulateEntryJob()
            {
                HashIndex = _hashIndex,
                Entries = entries
            };

            handle = depopulateJob.Schedule(_positions.Length, 128, handle);
            entries.Dispose(handle);
            
            var memsetCellStartJob = new MemsetCellStartJob()
            {
                CellStartEnd = _cellStartEnd.AsDeferredJobArray()
            };

            handle = memsetCellStartJob.Schedule(_cellStartEnd, 256, handle);
            
            var sortCellJob = new SortCellJob()
            {
                Positions = _positions,
                HashIndex = _hashIndex,
                CellStartEnd = _cellStartEnd.AsDeferredJobArray(),
                SortedPositions = _sortedPositions,
            };
            
            return sortCellJob.Schedule(handle);
        }
       
        public JobHandle UpdatePositions(Vector3[] newPositions)
        {
            NativeArray<float3> tempPositions = new NativeArray<Vector3>(newPositions, Allocator.TempJob).Reinterpret<float3>();
            var updateHandle = UpdatePositionsInternal(tempPositions);
            tempPositions.Dispose(updateHandle);
            return updateHandle;
        }

        public JobHandle UpdatePositions(NativeArray<float3> newPositions)
        {
            return UpdatePositionsInternal(newPositions);
        }
        
        private JobHandle UpdatePositionsInternal(NativeArray<float3> newPositions)
        {
            if (_positions.Length != newPositions.Length)
            {
                throw new Exception("Arrays are not the same length");
            }
            newPositions.CopyTo(_positions);
            return InitializeGridInternal();
        }
        
        public int[] SearchClosestPoint(Vector3[] queryPoints, bool checkSelf = false, float epsilon = 0.001f)
        {
            NativeArray<float3> qPoints = new NativeArray<Vector3>(queryPoints, Allocator.TempJob).Reinterpret<float3>();
            NativeArray<int> results = new NativeArray<int>(queryPoints.Length, Allocator.TempJob);
            
            var closestPointJob = new ClosestPointJob()
            {
                GridOrigin = _minMaxPositions,
                GridResolutionInv = 1.0f / _gridResolution,
                GridDimensions = _gridDimensions,
                QueryPositions = qPoints,
                SortedPositions = _sortedPositions,
                HashIndex = _hashIndex,
                CellStartEnd = _cellStartEnd.AsDeferredJobArray(),
                Results = results,
                IgnoreSelf = checkSelf,
                SquaredepsilonSelf = epsilon * epsilon,
            };

            var closestPointJobHandle = closestPointJob.Schedule(qPoints.Length, 16);
            int[] res = new int[qPoints.Length];
            
            qPoints.Dispose(closestPointJobHandle);
            closestPointJobHandle.Complete();

            results.CopyTo(res);
            results.Dispose();

            return res;
        }

        public NativeArray<int> SearchClosestPoint(NativeArray<float3> qPoints, bool checkSelf = false, float epsilon = 0.001f)
        {
            NativeArray<int> results = new NativeArray<int>(qPoints.Length, Allocator.TempJob);

            var closestPointJob = new ClosestPointJob()
            {
                GridOrigin = _minMaxPositions,
                GridResolutionInv = 1.0f / _gridResolution,
                GridDimensions = _gridDimensions,
                QueryPositions = qPoints,
                SortedPositions = _sortedPositions,
                HashIndex = _hashIndex,
                CellStartEnd = _cellStartEnd.AsDeferredJobArray(),
                Results = results,
                IgnoreSelf = checkSelf,
                SquaredepsilonSelf = epsilon * epsilon,
            };

            var closestPointJobHandle = closestPointJob.Schedule(qPoints.Length, 16);
            closestPointJobHandle.Complete();

            return results;
        }

        public int[] SearchWithin(Vector3[] queryPoints, float rad, int maxNeighborPerQuery)
        {
            NativeArray<float3> qPoints = new NativeArray<Vector3>(queryPoints, Allocator.TempJob).Reinterpret<float3>();
            NativeArray<int> results = new NativeArray<int>(queryPoints.Length * maxNeighborPerQuery, Allocator.TempJob);
            int cellsToLoop = (int)math.ceil(rad / _gridResolution);

            var withinJob = new FindWithinJob()
            {
                SquaredRadius = rad * rad,
                MaxNeighbor = maxNeighborPerQuery,
                CellsToLoop = cellsToLoop,
                GridOrigin = _minMaxPositions,
                invresoGrid = 1.0f / _gridResolution,
                GridDimensions = _gridDimensions,
                QueryPositions = qPoints,
                SortedPositions = _sortedPositions,
                HashIndex = _hashIndex,
                CellStartEnd = _cellStartEnd.AsDeferredJobArray(),
                Results = results
            };

            var withinJobHandle = withinJob.Schedule(qPoints.Length, 16);
            qPoints.Dispose(withinJobHandle);
            withinJobHandle.Complete();

            int[] res = new int[results.Length];
            results.CopyTo(res);
            results.Dispose();
            return res;
        }

        public NativeArray<int> SearchWithin(NativeArray<float3> queryPoints, float rad, int maxNeighborPerQuery)
        {
            NativeArray<int> results = new NativeArray<int>(queryPoints.Length * maxNeighborPerQuery, Allocator.TempJob);

            int cellsToLoop = (int)math.ceil(rad / _gridResolution);

            var withinJob = new FindWithinJob()
            {
                SquaredRadius = rad * rad,
                MaxNeighbor = maxNeighborPerQuery,
                CellsToLoop = cellsToLoop,
                GridOrigin = _minMaxPositions,
                invresoGrid = 1.0f / _gridResolution,
                GridDimensions = _gridDimensions,
                QueryPositions = queryPoints,
                SortedPositions = _sortedPositions,
                HashIndex = _hashIndex,
                CellStartEnd = _cellStartEnd.AsDeferredJobArray(),
                Results = results
            };

            var withinJobHandle = withinJob.Schedule(queryPoints.Length, 16);
            withinJobHandle.Complete();

            return results;
        }
        
        public void Dispose()
        {
            if (_positions.IsCreated)
                _positions.Dispose();
            if (_hashIndex.IsCreated)
                _hashIndex.Dispose();
            if (_cellStartEnd.IsCreated)
                _cellStartEnd.Dispose();
            if (_sortedPositions.IsCreated)
                _sortedPositions.Dispose();
            if (_minMaxPositions.IsCreated)
                _minMaxPositions.Dispose();
            if (_gridDimensions.IsCreated)
                _gridDimensions.Dispose();
        }
        
        [BurstCompile(CompileSynchronously = true)]
        struct GridInializationJob : IJob
        {
            [ReadOnly] public NativeArray<float3> Positions;
            [ReadOnly] public int TargetGridSize;
            public NativeArray<float3> MinMaxPositions;
            public NativeList<int2> CellStartEnd;
            public NativeArray<int3> GridDimensions;
            public float GridResolution;

            void IJob.Execute()
            {
                int N = Positions.Length;
                float3 minPosition = Positions[0];
                float3 maxPosition = Positions[0];
                float x, y, z;
                for (int i = 0; i < N; ++i)
                {
                    x = math.min(minPosition.x, Positions[i].x);
                    y = math.min(minPosition.y, Positions[i].y);
                    z = math.min(minPosition.z, Positions[i].z);
                    minPosition = new float3(x, y, z);
                    x = math.max(maxPosition.x, Positions[i].x);
                    y = math.max(maxPosition.y, Positions[i].y);
                    z = math.max(maxPosition.z, Positions[i].z);
                    maxPosition = new float3(x, y, z);
                }
                
                float3 sideLength = minPosition - maxPosition;
                float maxDist = math.max(sideLength.x, math.max(sideLength.y, sideLength.z));
                
                //Compute a resolution so that the grid is equal to 32*32*32 cells
                if (GridResolution <= 0.0f)
                {
                    GridResolution = maxDist / (float)TargetGridSize;
                }
                
                int gridSize = math.max(1, (int)math.ceil(maxDist / GridResolution));
                int3 gridDimension = new int3(gridSize, gridSize, gridSize);
                
                if (gridSize > MAXGRIDSIZE)
                {
                    throw new Exception("Grid is too large, adjust the resolution");
                }
                
                int cellCount = gridDimension.x * gridDimension.y * gridDimension.z;
                
                CellStartEnd.ResizeUninitialized(cellCount);
                GridDimensions[0] = gridDimension;
                MinMaxPositions[0] = minPosition;
                MinMaxPositions[1] = maxPosition;
            }
        }
        
        [BurstCompile(CompileSynchronously = true)]
        struct AssignHashJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> GridOrigin;
            [ReadOnly] public float GridResolutionInv;
            [ReadOnly] public NativeArray<int3> GridDimensions;
            [ReadOnly] public int CellCount;
            [ReadOnly] public NativeArray<float3> Positions;

            public NativeArray<int2> HashIndex;

            public void Execute(int index)
            {
                int3 dim = GridDimensions[0];
                float3 origin = GridOrigin[0];
                float3 p = Positions[index];
                int3 cell = SpaceToGrid(p, origin, GridResolutionInv);
                cell = math.clamp(cell, new int3(0, 0, 0), dim - new int3(1, 1, 1));
                int hash = Flatten3DTo1D(cell, dim);
                hash = math.clamp(hash, 0, CellCount - 1);

                HashIndex[index] = new (hash, index);
            }
        }
        [BurstCompile(CompileSynchronously = true)]
        struct MemsetCellStartJob : IJobParallelForDefer
        {
            public NativeArray<int2> CellStartEnd;

            void IJobParallelForDefer.Execute(int index)
            {
                int2 v;
                v.x = int.MaxValue - 1;
                v.y = int.MaxValue - 1;
                CellStartEnd[index] = v;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct SortCellJob : IJob
        {
            [ReadOnly] public NativeArray<float3> Positions;
            [ReadOnly] public NativeArray<int2> HashIndex;

            public NativeArray<int2> CellStartEnd;
            public NativeArray<float3> SortedPositions;

            void IJob.Execute()
            {
                for (int index = 0; index < HashIndex.Length; index++)
                {
                    int hash = HashIndex[index].x;
                    int id = HashIndex[index].y;
                    int2 newV;

                    int hashm1 = -1;
                    if (index != 0)
                        hashm1 = HashIndex[index - 1].x;


                    if (index == 0 || hash != hashm1)
                    {

                        newV.x = index;
                        newV.y = CellStartEnd[hash].y;

                        CellStartEnd[hash] = newV; // set start

                        if (index != 0)
                        {
                            newV.x = CellStartEnd[hashm1].x;
                            newV.y = index;
                            CellStartEnd[hashm1] = newV; // set end
                        }
                    }

                    if (index == Positions.Length - 1)
                    {
                        newV.x = CellStartEnd[hash].x;
                        newV.y = index + 1;

                        CellStartEnd[hash] = newV; // set end
                    }

                    // Reorder atoms according to sorted indices
                    SortedPositions[index] = Positions[id];
                }
            }
        }
        
        [BurstCompile(CompileSynchronously = true)]
        struct ClosestPointJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> GridOrigin;
            [ReadOnly] public float GridResolutionInv;
            [ReadOnly] public NativeArray<int3> GridDimensions;
            [ReadOnly] public NativeArray<float3> QueryPositions;
            [ReadOnly] public NativeArray<int2> CellStartEnd;
            [ReadOnly] public NativeArray<float3> SortedPositions;
            [ReadOnly] public NativeArray<int2> HashIndex;
            [ReadOnly] public bool IgnoreSelf;
            [ReadOnly] public float SquaredepsilonSelf;

            public NativeArray<int> Results;

            void IJobParallelFor.Execute(int index)
            {
                Results[index] = -1;
                float3 p = QueryPositions[index];

                float3 origin = GridOrigin[0];
                int3 dimensions = GridDimensions[0];
                int3 cell = SpaceToGrid(p, origin, GridResolutionInv);
                cell = math.clamp(cell, int3.zero, dimensions - new int3(1, 1, 1));

                float minD = float.MaxValue;
                int3 curGridId;
                int minRes = -1;

                for (int x = -1; x <= 1; x++)
                {
                    curGridId.x = cell.x + x;
                    if (curGridId.x >= 0 && curGridId.x < dimensions.x)
                    {
                        for (int y = -1; y <= 1; y++)
                        {
                            curGridId.y = cell.y + y;
                            if (curGridId.y >= 0 && curGridId.y < dimensions.y)
                            {
                                for (int z = -1; z <= 1; z++)
                                {
                                    curGridId.z = cell.z + z;
                                    if (curGridId.z >= 0 && curGridId.z < dimensions.z)
                                    {

                                        int neighcellhash = Flatten3DTo1D(curGridId, dimensions);
                                        
                                        int idStart = CellStartEnd[neighcellhash].x;
                                        int idStop = CellStartEnd[neighcellhash].y;

                                        if (idStart < int.MaxValue - 1)
                                        {
                                            for (int id = idStart; id < idStop; id++)
                                            {

                                                float3 posA = SortedPositions[id];
                                                float d = math.distancesq(p, posA); //Squared distance

                                                if (d < minD)
                                                {
                                                    if (IgnoreSelf)
                                                    {
                                                        if (d > SquaredepsilonSelf)
                                                        {
                                                            minRes = id;
                                                            minD = d;
                                                        }
                                                    }
                                                    else
                                                    {
                                                        minRes = id;
                                                        minD = d;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (minRes != -1)
                {
                    Results[index] = HashIndex[minRes].y;
                }
                else
                {
                    //Neighbor cells do not contain anything => compute all distances
                    //Compute all the distances = SLOW
                    // TODO: Improve this by progressively looping over outter layers or a growing cube
                    // TODO: This can also be done in another IJobParallel
                    for (int id = 0; id < SortedPositions.Length; id++)
                    {

                        float3 posA = SortedPositions[id];
                        float d = math.distancesq(p, posA);

                        if (d < minD)
                        {
                            if (IgnoreSelf)
                            {
                                if (d > SquaredepsilonSelf)
                                {
                                    minRes = id;
                                    minD = d;
                                }
                            }
                            else
                            {
                                minRes = id;
                                minD = d;
                            }
                        }
                    }
                    Results[index] = HashIndex[minRes].y;
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct FindWithinJob : IJobParallelFor
        {
            [ReadOnly] public float SquaredRadius;
            [ReadOnly] public int MaxNeighbor;
            [ReadOnly] public int CellsToLoop;
            [ReadOnly] public NativeArray<float3> GridOrigin;
            [ReadOnly] public float invresoGrid;
            [ReadOnly] public NativeArray<int3> GridDimensions;
            [ReadOnly] public NativeArray<float3> QueryPositions;
            [ReadOnly] public NativeArray<int2> CellStartEnd;
            [ReadOnly] public NativeArray<float3> SortedPositions;
            [ReadOnly] public NativeArray<int2> HashIndex;

            [NativeDisableParallelForRestriction]
            public NativeArray<int> Results;

            void IJobParallelFor.Execute(int index)
            {
                for (int i = 0; i < MaxNeighbor; i++)
                    Results[index * MaxNeighbor + i] = -1;

                float3 p = QueryPositions[index];

                int3 cell = SpaceToGrid(p, GridOrigin[0], invresoGrid);
                cell = math.clamp(cell, new int3(0, 0, 0), GridDimensions[0] - new int3(1, 1, 1));

                int3 curGridId;
                int idRes = 0;

                //First search for the corresponding cell
                int neighcellhashf = Flatten3DTo1D(cell, GridDimensions[0]);
                int idStartf = CellStartEnd[neighcellhashf].x;
                int idStopf = CellStartEnd[neighcellhashf].y;

                if (idStartf < int.MaxValue - 1)
                {
                    for (int id = idStartf; id < idStopf; id++)
                    {

                        float3 posA = SortedPositions[id];
                        float d = math.distancesq(p, posA); //Squared distance
                        if (d <= SquaredRadius)
                        {
                            Results[index * MaxNeighbor + idRes] = HashIndex[id].y;
                            idRes++;
                            //Found enough close points we can stop there
                            if (idRes == MaxNeighbor)
                            {
                                return;
                            }
                        }
                    }
                }

                for (int x = -CellsToLoop; x <= CellsToLoop; x++)
                {
                    curGridId.x = cell.x + x;
                    if (curGridId.x >= 0 && curGridId.x < GridDimensions[0].x)
                    {
                        for (int y = -CellsToLoop; y <= CellsToLoop; y++)
                        {
                            curGridId.y = cell.y + y;
                            if (curGridId.y >= 0 && curGridId.y < GridDimensions[0].y)
                            {
                                for (int z = -CellsToLoop; z <= CellsToLoop; z++)
                                {
                                    curGridId.z = cell.z + z;
                                    if (curGridId.z >= 0 && curGridId.z < GridDimensions[0].z)
                                    {
                                        if (x == 0 && y == 0 && z == 0)
                                            continue;//Already done that

                                        int neighcellhash = Flatten3DTo1D(curGridId, GridDimensions[0]);
                                        int idStart = CellStartEnd[neighcellhash].x;
                                        int idStop = CellStartEnd[neighcellhash].y;

                                        if (idStart < int.MaxValue - 1)
                                        {
                                            for (int id = idStart; id < idStop; id++)
                                            {

                                                float3 posA = SortedPositions[id];
                                                float d = math.distancesq(p, posA); //Squared distance

                                                if (d <= SquaredRadius)
                                                {
                                                    Results[index * MaxNeighbor + idRes] = HashIndex[id].y;
                                                    idRes++;
                                                    //Found enough close points we can stop there
                                                    if (idRes == MaxNeighbor)
                                                    {
                                                        return;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        public readonly struct SortEntry : IComparable<SortEntry>
        {
            public readonly int2 Value;
            public SortEntry(int2 value)
            {
                Value = value;
            }

            public int CompareTo(SortEntry other)
            {
                return Value.x.CompareTo(other.Value.x);
            }
        }
        
        [BurstCompile(CompileSynchronously = true)]
        private struct PopulateEntryJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<SortEntry> Entries;
            [ReadOnly] public NativeArray<int2> HashIndex;

            public void Execute(int index)
            {
                Entries[index] = new SortEntry(HashIndex[index]);
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        private struct DePopulateEntryJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<SortEntry> Entries;
            public NativeArray<int2> HashIndex;

            public void Execute(int index)
            {
                HashIndex[index] = Entries[index].Value;
            }
        }
        
        static int3 SpaceToGrid(float3 pos3D, float3 originGrid, float invdx)
        {
            return (int3)((pos3D - originGrid) * invdx);
        }
        static int Flatten3DTo1D(int3 id3d, int3 gridDim)
        {
            return (id3d.z * gridDim.x * gridDim.y) + (id3d.y * gridDim.x) + id3d.x;
        }
    }
}