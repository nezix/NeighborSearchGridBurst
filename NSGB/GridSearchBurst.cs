using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

//Multithreaded sort from https://coffeebraingames.wordpress.com/2020/06/07/a-multithreaded-sorting-attempt/

namespace BurstGridSearch
{

    public class GridSearchBurst
    {
        const int MAXGRIDSIZE = 256;

        NativeArray<float3> positions;
        NativeArray<float3> sortedPos;
        NativeArray<int2> hashIndex;
        NativeArray<int2> cellStartEnd;

        float3 minValue = float3.zero;
        float3 maxValue = float3.zero;
        int3 gridDim = int3.zero;

        float gridReso = -1.0f;
        int targetGridSize;

        public GridSearchBurst(float resolution, int targetGrid = 32)
        {
            if (resolution <= 0.0f && targetGrid > 0)
            {
                targetGridSize = targetGrid;
                return;
            }
            else if (resolution <= 0.0f && targetGrid <= 0)
            {
                throw new System.Exception("Wrong target grid size. Choose a resolution > 0 or a target grid > 0");
            }
            gridReso = resolution;
        }

        public void initGrid(Vector3[] pos)
        {

            positions = new NativeArray<Vector3>(pos, Allocator.Persistent).Reinterpret<float3>();

            _initGrid();
        }

        public void initGrid(NativeArray<float3> pos)
        {

            positions = new NativeArray<float3>(pos.Length, Allocator.Persistent);
            pos.CopyTo(positions);

            _initGrid();
        }

        private void _initGrid()
        {
            if (positions.Length == 0)
            {
                throw new System.Exception("Empty position buffer");
            }
            getMinMaxCoords(positions, ref minValue, ref maxValue);

            float3 sidelen = maxValue - minValue;
            float maxDist = math.max(sidelen.x, math.max(sidelen.y, sidelen.z));

            //Compute a resolution so that the grid is equal to 32*32*32 cells
            if (gridReso <= 0.0f)
            {
                gridReso = maxDist / (float)targetGridSize;
            }

            int gridSize = (int)math.ceil(maxDist / gridReso);
            gridDim = new int3(gridSize, gridSize, gridSize);

            if (gridSize > MAXGRIDSIZE)
            {
                throw new System.Exception("Grid is to large, adjust the resolution");
            }

            int NCells = gridDim.x * gridDim.y * gridDim.z;

            hashIndex = new NativeArray<int2>(positions.Length, Allocator.Persistent);
            sortedPos = new NativeArray<float3>(positions.Length, Allocator.Persistent);
            cellStartEnd = new NativeArray<int2>(NCells, Allocator.Persistent);

            var assignHashJob = new AssignHashJob()
            {
                oriGrid = minValue,
                invresoGrid = 1.0f / gridReso,
                gridDim = gridDim,
                pos = positions,
                nbcells = NCells,
                hashIndex = hashIndex
            };
            var assignHashJobHandle = assignHashJob.Schedule(positions.Length, 128);
            assignHashJobHandle.Complete();

            NativeArray<SortEntry> entries = new NativeArray<SortEntry>(positions.Length, Allocator.TempJob);

            var populateJob = new PopulateEntryJob()
            {
                hashIndex = hashIndex,
                entries = entries

            };
            var populateJobHandle = populateJob.Schedule(positions.Length, 128);
            populateJobHandle.Complete();


            // --- Here we could create a list for each filled cell of the grid instead of allocating the whole grid ---
            // hashIndex.Sort(new int2Comparer());//Sort by hash SUPER SLOW !

            // ------- Sort by hash

            JobHandle handle1 = new JobHandle();
            JobHandle chainHandle = MultithreadedSort.Sort(entries, handle1);
            chainHandle.Complete();
            handle1.Complete();

            var depopulateJob = new DePopulateEntryJob()
            {
                hashIndex = hashIndex,
                entries = entries
            };

            var depopulateJobHandle = depopulateJob.Schedule(positions.Length, 128);
            depopulateJobHandle.Complete();

            entries.Dispose();

            // ------- Sort (end)

            var memsetCellStartJob = new MemsetCellStartJob()
            {
                cellStartEnd = cellStartEnd
            };
            var memsetCellStartJobHandle = memsetCellStartJob.Schedule(NCells, 256);
            memsetCellStartJobHandle.Complete();

            var sortCellJob = new SortCellJob()
            {
                pos = positions,
                hashIndex = hashIndex,
                cellStartEnd = cellStartEnd,
                sortedPos = sortedPos
            };


            var sortCellJobHandle = sortCellJob.Schedule();
            sortCellJobHandle.Complete();
        }


        public void clean()
        {
            if (positions.IsCreated)
                positions.Dispose();
            if (hashIndex.IsCreated)
                hashIndex.Dispose();
            if (cellStartEnd.IsCreated)
                cellStartEnd.Dispose();
            if (sortedPos.IsCreated)
                sortedPos.Dispose();
        }

        public void updatePositions(Vector3[] newPos)
        {
            NativeArray<float3> tempPositions = new NativeArray<Vector3>(newPos, Allocator.TempJob).Reinterpret<float3>();
            updatePositions(tempPositions);
            tempPositions.Dispose();
        }

        ///Update the grid with new positions -> avoid allocating memory if not needed
        public void updatePositions(NativeArray<float3> newPos)
        {
            if (newPos.Length != positions.Length)
            {
                return;
            }

            newPos.CopyTo(positions);

            getMinMaxCoords(positions, ref minValue, ref maxValue);

            float3 sidelen = maxValue - minValue;
            float maxDist = math.max(sidelen.x, math.max(sidelen.y, sidelen.z));

            int gridSize = (int)math.ceil(maxDist / gridReso);
            gridDim = new int3(gridSize, gridSize, gridSize);

            if (gridSize > MAXGRIDSIZE)
            {
                throw new System.Exception("Grid is to large, adjust the resolution");
            }

            int NCells = gridDim.x * gridDim.y * gridDim.z;

            if (NCells != cellStartEnd.Length)
            {
                cellStartEnd.Dispose();
                cellStartEnd = new NativeArray<int2>(NCells, Allocator.Persistent);
            }
            var assignHashJob = new AssignHashJob()
            {
                oriGrid = minValue,
                invresoGrid = 1.0f / gridReso,
                gridDim = gridDim,
                pos = positions,
                nbcells = NCells,
                hashIndex = hashIndex
            };
            var assignHashJobHandle = assignHashJob.Schedule(positions.Length, 128);
            assignHashJobHandle.Complete();


            NativeArray<SortEntry> entries = new NativeArray<SortEntry>(positions.Length, Allocator.TempJob);

            var populateJob = new PopulateEntryJob()
            {
                hashIndex = hashIndex,
                entries = entries

            };
            var populateJobHandle = populateJob.Schedule(positions.Length, 128);
            populateJobHandle.Complete();


            // --- Here we could create a list for each filled cell of the grid instead of allocating the whole grid ---
            // hashIndex.Sort(new int2Comparer());//Sort by hash SUPER SLOW !

            // ------- Sort by hash

            JobHandle handle1 = new JobHandle();
            JobHandle chainHandle = MultithreadedSort.Sort(entries, handle1);
            chainHandle.Complete();
            handle1.Complete();

            var depopulateJob = new DePopulateEntryJob()
            {
                hashIndex = hashIndex,
                entries = entries

            };

            var depopulateJobHandle = depopulateJob.Schedule(positions.Length, 128);
            depopulateJobHandle.Complete();

            entries.Dispose();

            // ------- Sort (end)

            var memsetCellStartJob = new MemsetCellStartJob()
            {
                cellStartEnd = cellStartEnd
            };
            var memsetCellStartJobHandle = memsetCellStartJob.Schedule(NCells, 256);
            memsetCellStartJobHandle.Complete();

            var sortCellJob = new SortCellJob()
            {
                pos = positions,
                hashIndex = hashIndex,
                cellStartEnd = cellStartEnd,
                sortedPos = sortedPos
            };

            var sortCellJobHandle = sortCellJob.Schedule();
            sortCellJobHandle.Complete();
        }

        public int[] searchClosestPoint(Vector3[] queryPoints, bool checkSelf = false, float epsilon = 0.001f)
        {

            NativeArray<float3> qPoints = new NativeArray<Vector3>(queryPoints, Allocator.TempJob).Reinterpret<float3>();
            NativeArray<int> results = new NativeArray<int>(queryPoints.Length, Allocator.TempJob);

            var closestPointJob = new ClosestPointJob()
            {
                oriGrid = minValue,
                invresoGrid = 1.0f / gridReso,
                gridDim = gridDim,
                queryPos = qPoints,
                sortedPos = sortedPos,
                hashIndex = hashIndex,
                cellStartEnd = cellStartEnd,
                results = results,
                ignoreSelf = checkSelf,
                squaredepsilonSelf = epsilon * epsilon
            };

            var closestPointJobHandle = closestPointJob.Schedule(qPoints.Length, 16);
            closestPointJobHandle.Complete();

            int[] res = new int[qPoints.Length];
            results.CopyTo(res);

            qPoints.Dispose();
            results.Dispose();

            return res;
        }

        public NativeArray<int> searchClosestPoint(NativeArray<float3> qPoints, bool checkSelf = false, float epsilon = 0.001f)
        {

            NativeArray<int> results = new NativeArray<int>(qPoints.Length, Allocator.TempJob);

            var closestPointJob = new ClosestPointJob()
            {
                oriGrid = minValue,
                invresoGrid = 1.0f / gridReso,
                gridDim = gridDim,
                queryPos = qPoints,
                sortedPos = sortedPos,
                hashIndex = hashIndex,
                cellStartEnd = cellStartEnd,
                results = results,
                ignoreSelf = checkSelf,
                squaredepsilonSelf = epsilon * epsilon
            };

            var closestPointJobHandle = closestPointJob.Schedule(qPoints.Length, 16);
            closestPointJobHandle.Complete();

            return results;
        }

        public int[] searchWithin(Vector3[] queryPoints, float rad, int maxNeighborPerQuery)
        {

            NativeArray<float3> qPoints = new NativeArray<Vector3>(queryPoints, Allocator.TempJob).Reinterpret<float3>();
            NativeArray<int> results = new NativeArray<int>(queryPoints.Length * maxNeighborPerQuery, Allocator.TempJob);
            int cellsToLoop = (int)math.ceil(rad / gridReso);

            var withinJob = new FindWithinJob()
            {
                squaredRadius = rad * rad,
                maxNeighbor = maxNeighborPerQuery,
                cellsToLoop = cellsToLoop,
                oriGrid = minValue,
                invresoGrid = 1.0f / gridReso,
                gridDim = gridDim,
                queryPos = qPoints,
                sortedPos = sortedPos,
                hashIndex = hashIndex,
                cellStartEnd = cellStartEnd,
                results = results
            };

            var withinJobHandle = withinJob.Schedule(qPoints.Length, 16);
            withinJobHandle.Complete();

            int[] res = new int[results.Length];
            results.CopyTo(res);

            qPoints.Dispose();
            results.Dispose();

            return res;
        }

        public NativeArray<int> searchWithin(NativeArray<float3> queryPoints, float rad, int maxNeighborPerQuery)
        {

            NativeArray<int> results = new NativeArray<int>(queryPoints.Length * maxNeighborPerQuery, Allocator.TempJob);

            int cellsToLoop = (int)math.ceil(rad / gridReso);

            var withinJob = new FindWithinJob()
            {
                squaredRadius = rad * rad,
                maxNeighbor = maxNeighborPerQuery,
                cellsToLoop = cellsToLoop,
                oriGrid = minValue,
                invresoGrid = 1.0f / gridReso,
                gridDim = gridDim,
                queryPos = queryPoints,
                sortedPos = sortedPos,
                hashIndex = hashIndex,
                cellStartEnd = cellStartEnd,
                results = results
            };

            var withinJobHandle = withinJob.Schedule(queryPoints.Length, 16);
            withinJobHandle.Complete();

            return results;
        }

        //---------------------------------------------

        void getMinMaxCoords(NativeArray<float3> mpos, ref float3 minV, ref float3 maxV)
        {
            NativeArray<float3> tmpmin = new NativeArray<float3>(1, Allocator.TempJob);
            NativeArray<float3> tmpmax = new NativeArray<float3>(1, Allocator.TempJob);
            var mmJob = new getminmaxJob()
            {
                minVal = tmpmin,
                maxVal = tmpmax,
                pos = mpos
            };
            var mmJobHandle = mmJob.Schedule(mpos.Length, new JobHandle());
            mmJobHandle.Complete();
            minV = tmpmin[0];
            maxV = tmpmax[0];
            tmpmin.Dispose();
            tmpmax.Dispose();
        }


        [BurstCompile(CompileSynchronously = true)]
        struct getminmaxJob : IJobFor
        {
            public NativeArray<float3> minVal;
            public NativeArray<float3> maxVal;
            [ReadOnly] public NativeArray<float3> pos;

            void IJobFor.Execute(int i)
            {
                float x, y, z;
                if (i == 0)
                {
                    minVal[0] = pos[0];
                    maxVal[0] = pos[0];
                }
                else
                {

                    x = math.min(minVal[0].x, pos[i].x);
                    y = math.min(minVal[0].y, pos[i].y);
                    z = math.min(minVal[0].z, pos[i].z);
                    minVal[0] = new float3(x, y, z);
                    x = math.max(maxVal[0].x, pos[i].x);
                    y = math.max(maxVal[0].y, pos[i].y);
                    z = math.max(maxVal[0].z, pos[i].z);
                    maxVal[0] = new float3(x, y, z);
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct AssignHashJob : IJobParallelFor
        {
            [ReadOnly] public float3 oriGrid;
            [ReadOnly] public float invresoGrid;
            [ReadOnly] public int3 gridDim;
            [ReadOnly] public int nbcells;
            [ReadOnly] public NativeArray<float3> pos;
            public NativeArray<int2> hashIndex;

            void IJobParallelFor.Execute(int index)
            {
                float3 p = pos[index];

                int3 cell = spaceToGrid(p, oriGrid, invresoGrid);
                cell = math.clamp(cell, new int3(0, 0, 0), gridDim - new int3(1, 1, 1));
                int hash = flatten3DTo1D(cell, gridDim);
                hash = math.clamp(hash, 0, nbcells - 1);

                int2 v;
                v.x = hash;
                v.y = index;

                hashIndex[index] = v;
            }
        }


        [BurstCompile(CompileSynchronously = true)]
        struct MemsetCellStartJob : IJobParallelFor
        {
            public NativeArray<int2> cellStartEnd;

            void IJobParallelFor.Execute(int index)
            {
                int2 v;
                v.x = int.MaxValue - 1;
                v.y = int.MaxValue - 1;
                cellStartEnd[index] = v;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct SortCellJob : IJob
        {

            [ReadOnly] public NativeArray<float3> pos;
            [ReadOnly] public NativeArray<int2> hashIndex;

            public NativeArray<int2> cellStartEnd;

            public NativeArray<float3> sortedPos;

            void IJob.Execute()
            {
                for (int index = 0; index < hashIndex.Length; index++)
                {
                    int hash = hashIndex[index].x;
                    int id = hashIndex[index].y;
                    int2 newV;

                    int hashm1 = -1;
                    if (index != 0)
                        hashm1 = hashIndex[index - 1].x;


                    if (index == 0 || hash != hashm1)
                    {

                        newV.x = index;
                        newV.y = cellStartEnd[hash].y;

                        cellStartEnd[hash] = newV; // set start

                        if (index != 0)
                        {
                            newV.x = cellStartEnd[hashm1].x;
                            newV.y = index;
                            cellStartEnd[hashm1] = newV; // set end
                        }
                    }

                    if (index == pos.Length - 1)
                    {
                        newV.x = cellStartEnd[hash].x;
                        newV.y = index + 1;

                        cellStartEnd[hash] = newV; // set end
                    }

                    // Reorder atoms according to sorted indices
                    sortedPos[index] = pos[id];
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct ClosestPointJob : IJobParallelFor
        {
            [ReadOnly] public float3 oriGrid;
            [ReadOnly] public float invresoGrid;
            [ReadOnly] public int3 gridDim;
            [ReadOnly] public NativeArray<float3> queryPos;
            [ReadOnly] public NativeArray<int2> cellStartEnd;
            [ReadOnly] public NativeArray<float3> sortedPos;
            [ReadOnly] public NativeArray<int2> hashIndex;
            [ReadOnly] public bool ignoreSelf;
            [ReadOnly] public float squaredepsilonSelf;

            public NativeArray<int> results;

            void IJobParallelFor.Execute(int index)
            {
                results[index] = -1;
                float3 p = queryPos[index];

                int3 cell = spaceToGrid(p, oriGrid, invresoGrid);
                cell = math.clamp(cell, new int3(0, 0, 0), gridDim - new int3(1, 1, 1));

                float minD = float.MaxValue;
                int3 curGridId;
                int minRes = -1;


                cell = math.clamp(cell, new int3(0, 0, 0), gridDim - new int3(1, 1, 1));


                int neighcellhashf = flatten3DTo1D(cell, gridDim);
                int idStartf = cellStartEnd[neighcellhashf].x;
                int idStopf = cellStartEnd[neighcellhashf].y;

                if (idStartf < int.MaxValue - 1)
                {
                    for (int id = idStartf; id < idStopf; id++)
                    {

                        float3 posA = sortedPos[id];
                        float d = math.distancesq(p, posA); //Squared distance

                        if (d < minD)
                        {
                            if (ignoreSelf)
                            {
                                if (d > squaredepsilonSelf)
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

                if (minRes != -1)
                {
                    results[index] = hashIndex[minRes].y;
                    return;
                }

                //Corresponding cell was empty, let's search in neighbor cells
                for (int x = -1; x <= 1; x++)
                {
                    curGridId.x = cell.x + x;
                    if (curGridId.x >= 0 && curGridId.x < gridDim.x)
                    {
                        for (int y = -1; y <= 1; y++)
                        {
                            curGridId.y = cell.y + y;
                            if (curGridId.y >= 0 && curGridId.y < gridDim.y)
                            {
                                for (int z = -1; z <= 1; z++)
                                {
                                    curGridId.z = cell.z + z;
                                    if (curGridId.z >= 0 && curGridId.z < gridDim.z)
                                    {

                                        int neighcellhash = flatten3DTo1D(curGridId, gridDim);
                                        int idStart = cellStartEnd[neighcellhash].x;
                                        int idStop = cellStartEnd[neighcellhash].y;

                                        if (idStart < int.MaxValue - 1)
                                        {
                                            for (int id = idStart; id < idStop; id++)
                                            {

                                                float3 posA = sortedPos[id];
                                                float d = math.distancesq(p, posA); //Squared distance

                                                if (d < minD)
                                                {
                                                    if (ignoreSelf)
                                                    {
                                                        if (d > squaredepsilonSelf)
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
                    results[index] = hashIndex[minRes].y;
                }
                else
                {//Neighbor cells do not contain anything => compute all distances
                 //Compute all the distances ! = SLOW
                    for (int id = 0; id < sortedPos.Length; id++)
                    {

                        float3 posA = sortedPos[id];
                        float d = math.distancesq(p, posA); //Squared distance

                        if (d < minD)
                        {
                            if (ignoreSelf)
                            {
                                if (d > squaredepsilonSelf)
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
                    results[index] = hashIndex[minRes].y;
                }
            }
        }



        [BurstCompile(CompileSynchronously = true)]
        struct FindWithinJob : IJobParallelFor
        {
            [ReadOnly] public float squaredRadius;
            [ReadOnly] public int maxNeighbor;
            [ReadOnly] public int cellsToLoop;
            [ReadOnly] public float3 oriGrid;
            [ReadOnly] public float invresoGrid;
            [ReadOnly] public int3 gridDim;
            [ReadOnly] public NativeArray<float3> queryPos;
            [ReadOnly] public NativeArray<int2> cellStartEnd;
            [ReadOnly] public NativeArray<float3> sortedPos;
            [ReadOnly] public NativeArray<int2> hashIndex;

            [NativeDisableParallelForRestriction]
            public NativeArray<int> results;

            void IJobParallelFor.Execute(int index)
            {
                for (int i = 0; i < maxNeighbor; i++)
                    results[index * maxNeighbor + i] = -1;

                float3 p = queryPos[index];

                int3 cell = spaceToGrid(p, oriGrid, invresoGrid);
                cell = math.clamp(cell, new int3(0, 0, 0), gridDim - new int3(1, 1, 1));

                int3 curGridId;
                int idRes = 0;

                //First search for the corresponding cell
                int neighcellhashf = flatten3DTo1D(cell, gridDim);
                int idStartf = cellStartEnd[neighcellhashf].x;
                int idStopf = cellStartEnd[neighcellhashf].y;


                if (idStartf < int.MaxValue - 1)
                {
                    for (int id = idStartf; id < idStopf; id++)
                    {

                        float3 posA = sortedPos[id];
                        float d = math.distancesq(p, posA); //Squared distance
                        if (d <= squaredRadius)
                        {
                            results[index * maxNeighbor + idRes] = hashIndex[id].y;
                            idRes++;
                            //Found enough close points we can stop there
                            if (idRes == maxNeighbor)
                            {
                                return;
                            }
                        }
                    }
                }

                for (int x = -cellsToLoop; x <= cellsToLoop; x++)
                {
                    curGridId.x = cell.x + x;
                    if (curGridId.x >= 0 && curGridId.x < gridDim.x)
                    {
                        for (int y = -cellsToLoop; y <= cellsToLoop; y++)
                        {
                            curGridId.y = cell.y + y;
                            if (curGridId.y >= 0 && curGridId.y < gridDim.y)
                            {
                                for (int z = -cellsToLoop; z <= cellsToLoop; z++)
                                {
                                    curGridId.z = cell.z + z;
                                    if (curGridId.z >= 0 && curGridId.z < gridDim.z)
                                    {
                                        if (x == 0 && y == 0 && z == 0)
                                            continue;//Already done that


                                        int neighcellhash = flatten3DTo1D(curGridId, gridDim);
                                        int idStart = cellStartEnd[neighcellhash].x;
                                        int idStop = cellStartEnd[neighcellhash].y;

                                        if (idStart < int.MaxValue - 1)
                                        {
                                            for (int id = idStart; id < idStop; id++)
                                            {

                                                float3 posA = sortedPos[id];
                                                float d = math.distancesq(p, posA); //Squared distance

                                                if (d <= squaredRadius)
                                                {
                                                    results[index * maxNeighbor + idRes] = hashIndex[id].y;
                                                    idRes++;
                                                    //Found enough close points we can stop there
                                                    if (idRes == maxNeighbor)
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

        //--------- Fast sort stuff
        [BurstCompile(CompileSynchronously = true)]
        private struct PopulateEntryJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<SortEntry> entries;
            [ReadOnly] public NativeArray<int2> hashIndex;

            public void Execute(int index)
            {
                this.entries[index] = new SortEntry(hashIndex[index]);
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        private struct DePopulateEntryJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<SortEntry> entries;
            public NativeArray<int2> hashIndex;

            public void Execute(int index)
            {
                hashIndex[index] = entries[index].value;
            }
        }

        public struct int2Comparer : IComparer<int2>
        {
            public int Compare(int2 lhs, int2 rhs)
            {
                return lhs.x.CompareTo(rhs.x);
            }
        }

        static int3 spaceToGrid(float3 pos3D, float3 originGrid, float invdx)
        {
            return (int3)((pos3D - originGrid) * invdx);
        }
        static int flatten3DTo1D(int3 id3d, int3 gridDim)
        {
            return (id3d.z * gridDim.x * gridDim.y) + (id3d.y * gridDim.x) + id3d.x;
            // return (gridDim.y * gridDim.z * id3d.x) + (gridDim.z * id3d.y) + id3d.z;
        }


        public static class ConcreteJobs
        {
            static ConcreteJobs()
            {
                new MultithreadedSort.Merge<SortEntry>().Schedule();
                new MultithreadedSort.QuicksortJob<SortEntry>().Schedule();
            }
        }

        // This is the item to sort
        public readonly struct SortEntry : IComparable<SortEntry>
        {
            public readonly int2 value;

            public SortEntry(int2 value)
            {
                this.value = value;
            }

            public int CompareTo(SortEntry other)
            {
                return this.value.x.CompareTo(other.value.x);
            }
        }
    }

    public static class MultithreadedSort
    {
        // Use quicksort when sub-array length is less than or equal than this value
        public const int QUICKSORT_THRESHOLD_LENGTH = 400;

        public static JobHandle Sort<T>(NativeArray<T> array, JobHandle parentHandle)
        where T : unmanaged, IComparable<T>
        {
            return MergeSort(array, new SortRange(0, array.Length - 1), parentHandle);
        }

        // public static JobHandle Sort<T>(NativeArray<T> array, JobHandle parentHandle)
        // where T : unmanaged, IComparable<T> {
        //     return NativeSortExtension.SortJob(array, parentHandle);
        // }


        private static JobHandle MergeSort<T>(NativeArray<T> array, SortRange range, JobHandle parentHandle) where T : unmanaged, IComparable<T>
        {
            if (range.Length <= QUICKSORT_THRESHOLD_LENGTH)
            {
                // Use quicksort
                return new QuicksortJob<T>()
                {
                    array = array,
                    left = range.left,
                    right = range.right
                }.Schedule(parentHandle);
            }

            int middle = range.Middle;

            SortRange left = new SortRange(range.left, middle);
            JobHandle leftHandle = MergeSort(array, left, parentHandle);

            SortRange right = new SortRange(middle + 1, range.right);
            JobHandle rightHandle = MergeSort(array, right, parentHandle);

            JobHandle combined = JobHandle.CombineDependencies(leftHandle, rightHandle);

            return new Merge<T>()
            {
                array = array,
                first = left,
                second = right
            }.Schedule(combined);
        }

        public readonly struct SortRange
        {
            public readonly int left;
            public readonly int right;

            public SortRange(int left, int right)
            {
                this.left = left;
                this.right = right;
            }

            public int Length
            {
                get
                {
                    return this.right - this.left + 1;
                }
            }

            public int Middle
            {
                get
                {
                    return (this.left + this.right) >> 1; // divide 2
                }
            }

            public int Max
            {
                get
                {
                    return this.right;
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        public struct Merge<T> : IJob where T : unmanaged, IComparable<T>
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<T> array;

            public SortRange first;
            public SortRange second;

            public void Execute()
            {
                int firstIndex = this.first.left;
                int secondIndex = this.second.left;
                int resultIndex = this.first.left;

                // Copy first
                NativeArray<T> copy = new NativeArray<T>(this.second.right - this.first.left + 1, Allocator.Temp);
                for (int i = this.first.left; i <= this.second.right; ++i)
                {
                    int copyIndex = i - this.first.left;
                    copy[copyIndex] = this.array[i];
                }

                while (firstIndex <= this.first.Max || secondIndex <= this.second.Max)
                {
                    if (firstIndex <= this.first.Max && secondIndex <= this.second.Max)
                    {
                        // both subranges still have elements
                        T firstValue = copy[firstIndex - this.first.left];
                        T secondValue = copy[secondIndex - this.first.left];

                        if (firstValue.CompareTo(secondValue) < 0)
                        {
                            // first value is lesser
                            this.array[resultIndex] = firstValue;
                            ++firstIndex;
                            ++resultIndex;
                        }
                        else
                        {
                            this.array[resultIndex] = secondValue;
                            ++secondIndex;
                            ++resultIndex;
                        }
                    }
                    else if (firstIndex <= this.first.Max)
                    {
                        // Only the first range has remaining elements
                        T firstValue = copy[firstIndex - this.first.left];
                        this.array[resultIndex] = firstValue;
                        ++firstIndex;
                        ++resultIndex;
                    }
                    else if (secondIndex <= this.second.Max)
                    {
                        // Only the second range has remaining elements
                        T secondValue = copy[secondIndex - this.first.left];
                        this.array[resultIndex] = secondValue;
                        ++secondIndex;
                        ++resultIndex;
                    }
                }

                copy.Dispose();
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        public struct QuicksortJob<T> : IJob where T : unmanaged, IComparable<T>
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<T> array;

            public int left;
            public int right;

            public void Execute()
            {
                Quicksort(this.left, this.right);
            }

            private void Quicksort(int left, int right)
            {
                int i = left;
                int j = right;
                T pivot = this.array[(left + right) / 2];

                while (i <= j)
                {
                    // Lesser
                    while (this.array[i].CompareTo(pivot) < 0)
                    {
                        ++i;
                    }

                    // Greater
                    while (this.array[j].CompareTo(pivot) > 0)
                    {
                        --j;
                    }

                    if (i <= j)
                    {
                        // Swap
                        T temp = this.array[i];
                        this.array[i] = this.array[j];
                        this.array[j] = temp;

                        ++i;
                        --j;
                    }
                }

                // Recurse
                if (left < j)
                {
                    Quicksort(left, j);
                }

                if (i < right)
                {
                    Quicksort(i, right);
                }
            }
        }
    }
}
