using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

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
        private NativeArray<int2> _cellStartEnd;
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
            positions.CopyTo(_positions);
            return InitializeGridInternal();
        }
        
        public JobHandle InitializeGrid(Vector3[] positions)
        {
            Dispose();
            _positions = new NativeArray<Vector3>(positions, Allocator.Persistent).Reinterpret<float3>();
            return InitializeGridInternal();
        }
        
        private JobHandle InitializeGridInternal()
        {
            if (_positions.Length == 0)
            {
                throw new Exception("Empty position buffer");
            }            
            
            NativeList<int2> hashIndex = new (Allocator.Persistent);
            NativeList<float3> sortedPositions = new (Allocator.Persistent);
            NativeList<int2> cellStartEnd = new (Allocator.Persistent);
            _minMaxPositions = new (2, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            _gridDimensions = new (1, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            
            var job = new GridInializationJob()
            {
                Positions = _positions,
                GridResolution = _gridResolution,
                TargetGridSize = _targetGridSize,
                HashIndex = hashIndex,
                SortedPositions = sortedPositions, 
                CellStartEnd = cellStartEnd,
                MinMaxPositions = _minMaxPositions,
            };
            var handle = job.Schedule();

            var assignHashJob = new AssignHashJob()
            {
                GridOrigin = _minMaxPositions,
                GridResolutionInv = 1.0f / _gridResolution,
                GridDimensions = _gridDimensions,
                Positions = _positions,
                HashIndex = hashIndex.AsDeferredJobArray(),
                CellCount = cellStartEnd.AsDeferredJobArray().Length,
            };
            handle = assignHashJob.Schedule(_positions.Length, 128, handle);
            
            NativeArray<SortEntry> entries = new NativeArray<SortEntry>(_positions.Length, Allocator.TempJob);
            
            var populateJob = new PopulateEntryJob()
            {
                HashIndex = hashIndex.AsDeferredJobArray(),
                Entries = entries
            };
            
            handle = populateJob.Schedule(_positions.Length, 128, handle);
            
            // ------- Sort by hash
            handle = MultithreadedSort.Sort(entries, handle);
            
            var depopulateJob = new DePopulateEntryJob()
            {
                HashIndex = hashIndex.AsDeferredJobArray(),
                Entries = entries
            };

            handle = depopulateJob.Schedule(_positions.Length, 128, handle);
            entries.Dispose(handle);
            
            var memsetCellStartJob = new MemsetCellStartJob()
            {
                CellStartEnd = cellStartEnd.AsDeferredJobArray()
            };

            handle = memsetCellStartJob.Schedule(cellStartEnd, 256, handle);
            
            var sortCellJob = new SortCellJob()
            {
                Positions = _positions,
                HashIndex = hashIndex.AsDeferredJobArray(),
                CellStartEnd = cellStartEnd.AsDeferredJobArray(),
                SortedPositions = sortedPositions.AsDeferredJobArray(),
            };
            
            _cellStartEnd = cellStartEnd.AsDeferredJobArray();
            _sortedPositions = sortedPositions.AsDeferredJobArray();
            _hashIndex = hashIndex.AsDeferredJobArray();
            
            return sortCellJob.Schedule(handle);
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
            public NativeList<int2> HashIndex;
            public NativeList<int2> CellStartEnd;
            public NativeList<float3> SortedPositions;
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
                HashIndex.ResizeUninitialized(N);
                SortedPositions.ResizeUninitialized(N);

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