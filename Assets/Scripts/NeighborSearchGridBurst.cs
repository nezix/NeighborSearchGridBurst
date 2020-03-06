using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using System.Linq;


namespace Nezix {
public class NeighborSearchGridBurst {

	const int MAXCELLS = 5000000;//~170^3
	public bool isInit = false;
	private NativeArray<float3> sortedPos;
	private NativeArray<int2> cellStartEnd;
	private NativeArray<int2> hashIndex;

	public float3 oriGrid;
	public float resoGrid;
	public int3 gridDim;


	///Can contain duplicates
	public NativeArray<int> getPointsInRadius(NativeArray<float3> allPoints, NativeArray<float3> qPoints, int maxRes, float cutoff) {
		NativeArray<int> results = new NativeArray<int>(maxRes * qPoints.Length, Allocator.Persistent);

		initGrid(allPoints);

		radiusSearch(qPoints, results, maxRes, cutoff);

		clear();
		return results;
	}



	void initGrid(NativeArray<float3> positions, float gridReso = 0.5f) {

		if (positions.Length < 3) {
			Debug.LogError("Too few points to search");
			return;
		}

		float3 minValue = float3.zero;
		float3 maxValue = float3.zero;

		getMinMaxCoords(positions, ref minValue, ref maxValue);

		float3 originGrid = minValue;
		float gridResolutionNeighbor = gridReso ;
		//TODO Find a "good" resolution, for example:
		//Sample X close points and find the mean distance

		float maxDist = math.max(maxValue.x - minValue.x, math.max(maxValue.y - minValue.y, maxValue.z - minValue.z));

		oriGrid = originGrid;
		resoGrid = gridResolutionNeighbor;

		if (maxDist < 0.1f) {
			Debug.LogError("Failed to init grid for neighbor search");
			return;
		}

		int gridNeighborSize = (int)math.ceil(maxDist / gridResolutionNeighbor);
		int3 gridNeighborDim = new int3(gridNeighborSize, gridNeighborSize, gridNeighborSize);
		int nbcellsNeighbor = gridNeighborDim.x * gridNeighborDim.y * gridNeighborDim.z;

		Debug.Log(gridNeighborDim.x + " x " + gridNeighborDim.y + " x " + gridNeighborDim.z + " = " + nbcellsNeighbor);
		if (nbcellsNeighbor > MAXCELLS) {
			Debug.LogError(gridNeighborDim.x + " x " + gridNeighborDim.y + " x " + gridNeighborDim.z +
			               " = " + nbcellsNeighbor + " => Grid is too large, try changing the grid resolution");
			return;
		}

		gridDim = gridNeighborDim;

		hashIndex = new NativeArray<int2>(positions.Length, Allocator.TempJob);
		sortedPos = new NativeArray<float3>(positions.Length, Allocator.Persistent);
		cellStartEnd = new NativeArray<int2>(nbcellsNeighbor, Allocator.Persistent);

		//Assign a cell id to each point

		var assignHashJob = new AssignHashJob() {
			oriGrid = originGrid,
			resoGrid = gridResolutionNeighbor,
			gridDim = gridNeighborDim,
			pos = positions,
			hashIndex = hashIndex
		};
		var assignHashJobHandle = assignHashJob.Schedule(positions.Length, 128);
		assignHashJobHandle.Complete();

		//Sort the points based on the cell id
		hashIndex.Sort(new int2Comparer());//Compare int2 !


		//Fill the grid with empty cells
		var memsetCellStartJob = new MemsetCellStartJob() {
			cellStartEnd = cellStartEnd
		};
		var memsetCellStartJobHandle = memsetCellStartJob.Schedule(nbcellsNeighbor, 256);
		memsetCellStartJobHandle.Complete();

		//Fill non-empty cells with point indices and reorder the point array
		var sortCellJob = new SortCellJob() {
			pos = positions,
			hashIndex = hashIndex,
			cellStartEnd = cellStartEnd,
			sortedPos = sortedPos
		};

		var sortCellJobHandle = sortCellJob.Schedule(positions.Length, 128);
		sortCellJobHandle.Complete();

		isInit = true;
	}

	public void clear() {
		sortedPos.Dispose();
		cellStartEnd.Dispose();
		hashIndex.Dispose();
	}


	public void radiusSearch(NativeArray<float3> qPoints, NativeArray<int> results, int maxRes, float cutoff) {
		float start = Time.realtimeSinceStartup;

		var radiusJob = new RadiusSearchJob() {
			oriGrid = oriGrid,
			resoGrid = resoGrid,
			gridDim = gridDim,
			queryPos = qPoints,
			sortedPos = sortedPos,
			cellStartEnd = cellStartEnd,
			radius = cutoff,
			radrad = cutoff * cutoff,
			maxRes = maxRes,
			results = results,
			hashIndex = hashIndex
		};
		var radiusJobHandle = radiusJob.Schedule(qPoints.Length, 64);
		radiusJobHandle.Complete();

		results.Sort(new NeighborSearchGridBurst.intInvComparer());


		Debug.Log("Time for grid search: " + (1000.0f * (Time.realtimeSinceStartup - start)).ToString("f3") + " ms");
	}


	void getMinMaxCoords(NativeArray<float3> positions, ref float3 minV, ref float3 maxV) {
		NativeArray<float3> tmpmin = new NativeArray<float3>(1, Allocator.TempJob);
		NativeArray<float3> tmpmax = new NativeArray<float3>(1, Allocator.TempJob);
		var mmJob = new getminmaxJob() {
			minVal = tmpmin,
			maxVal = tmpmax,
			pos = positions
		};
		var mmJobHandle = mmJob.Schedule();
		mmJobHandle.Complete();
		minV = tmpmin[0];
		maxV = tmpmax[0];
		tmpmin.Dispose();
		tmpmax.Dispose();
	}


	[BurstCompile]
	struct getminmaxJob : IJob {
		public NativeArray<float3> minVal;
		public NativeArray<float3> maxVal;
		[ReadOnly] public NativeArray<float3> pos;

		void IJob.Execute() {
			minVal[0] = pos[0];
			maxVal[0] = pos[0];
			float x, y, z;
			for (int i = 1; i < pos.Length; i++) {
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


	[BurstCompile]
	struct AssignHashJob : IJobParallelFor {
		[ReadOnly] public float3 oriGrid;
		[ReadOnly] public float resoGrid;
		[ReadOnly] public int3 gridDim;
		[ReadOnly] public NativeArray<float3> pos;
		public NativeArray<int2> hashIndex;

		void IJobParallelFor.Execute(int index)
		{
			float3 p = pos[index];

			int3 cell = spaceToGrid(p, oriGrid, resoGrid);
			int hash = flatten3DTo1D(cell, gridDim);

			int2 v;
			v.x = hash;
			v.y = index;
			hashIndex[index] = v;
		}
		int3 spaceToGrid(float3 pos3D, float3 originGrid, float dx) {
			return (int3)((pos3D - originGrid) / dx);
		}
		int flatten3DTo1D(int3 id3d, int3 gridDim) {
			return (gridDim.y * gridDim.z * id3d.x) + (gridDim.z * id3d.y) + id3d.z;
		}
	}

	[BurstCompile]
	struct MemsetCellStartJob : IJobParallelFor {
		public NativeArray<int2> cellStartEnd;

		void IJobParallelFor.Execute(int index)
		{
			int2 v;
			v.x = int.MaxValue - 1;
			v.y = int.MaxValue - 1;
			cellStartEnd[index] = v ;
		}
	}

	[BurstCompile]
	struct SortCellJob : IJobParallelFor {

		[ReadOnly] public NativeArray<float3> pos;
		[ReadOnly] public NativeArray<int2> hashIndex;

		[NativeDisableParallelForRestriction]
		public NativeArray<int2> cellStartEnd;

		public NativeArray<float3> sortedPos;


		void IJobParallelFor.Execute(int index)
		{
			int hash = hashIndex[index].x;
			int id = hashIndex[index].y;
			int2 newV;

			int hashm1;
			if (index != 0)
				hashm1 = hashIndex[index - 1].x;
			else
				hashm1 = hash;


			if (index == 0 || hash != hashm1) {
				newV.x = index;
				newV.y = cellStartEnd[hash].y;

				cellStartEnd[hash] = newV; // set start

				if (index > 0) {
					newV.x = cellStartEnd[hashm1].x;
					newV.y = index;
					cellStartEnd[hashm1] = newV; // set end
				}
			}

			if (index == pos.Length - 1) {
				newV.x = cellStartEnd[hash].x;
				newV.y = index + 1;

				cellStartEnd[hash] = newV; // set end
			}

			// Reorder atoms according to sorted indices

			sortedPos[index] = pos[id];
		}
	}


	[BurstCompile]
	struct RadiusSearchJob : IJobParallelFor {
		[ReadOnly] public float3 oriGrid;
		[ReadOnly] public float resoGrid;
		[ReadOnly] public int3 gridDim;
		[ReadOnly] public NativeArray<float3> queryPos;
		[ReadOnly] public NativeArray<int2> cellStartEnd;
		[ReadOnly] public NativeArray<float3> sortedPos;
		[ReadOnly] public float radius;
		[ReadOnly] public float radrad;
		[ReadOnly] public NativeArray<int2> hashIndex;
		[ReadOnly] public int maxRes;

		[NativeDisableParallelForRestriction]
		public NativeArray<int> results;

		void IJobParallelFor.Execute(int index) {

			for (int i = 0; i < maxRes; i++) {
				results[index * maxRes + i] = -1;
			}

			int curId = 0;
			float3 p = queryPos[index];
			int3 cell = spaceToGrid(p, oriGrid, resoGrid);

			int3 curGridId;
			int range = math.max(1, (int)(radius / resoGrid));

			for (int x = -range; x <= range; x++) {
				curGridId.x = cell.x + x;
				if (curGridId.x >= 0 && curGridId.x < gridDim.x) {
					for (int y = -range; y <= range; y++) {
						curGridId.y = cell.y + y;
						if (curGridId.y >= 0 && curGridId.y < gridDim.y) {
							for (int z = -range; z <= range; z++) {
								curGridId.z = cell.z + z;
								if (curGridId.z >= 0 && curGridId.z < gridDim.z) {

									int neighcellhash = flatten3DTo1D(curGridId, gridDim);
									int idStart = cellStartEnd[neighcellhash].x;
									int idStop = cellStartEnd[neighcellhash].y;

									if (idStart < int.MaxValue - 1) {//Not empty cell
										for (int id = idStart; id < idStop; id++) {

											float3 posA = sortedPos[id];
											float d = sqr_distance(posA, p);
											// Debug.Log(curGridId + " / "+id+" / "+math.sqrt(d));
											if (d <= radrad) {
												if(curId < maxRes){
													results[index * maxRes + curId] = hashIndex[id].y;
													curId++;
												}
												else{
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
		int3 spaceToGrid(float3 pos3D, float3 originGrid, float dx) {
			return (int3)((pos3D - originGrid) / dx);
		}
		int flatten3DTo1D(int3 id3d, int3 gridDim) {
			return (gridDim.y * gridDim.z * id3d.x) + (gridDim.z * id3d.y) + id3d.z;
		}
		float sqr_distance(float3 p1, float3 p2) {
			float x = (p1.x - p2.x) * (p1.x - p2.x);
			float y = (p1.y - p2.y) * (p1.y - p2.y);
			float z = (p1.z - p2.z) * (p1.z - p2.z);

			return x + y + z;
		}
	}




	// [BurstCompile]
	// struct ClosestPointJob : IJobParallelFor {
	// 	[ReadOnly] public float3 oriGrid;
	// 	[ReadOnly] public float resoGrid;
	// 	[ReadOnly] public int3 gridDim;
	// 	[ReadOnly] public NativeArray<float3> queryPos;
	// 	[ReadOnly] public NativeArray<int2> cellStartEnd;
	// 	[ReadOnly] public NativeArray<float3> sortedPos;
	// 	public NativeArray<int> results;

	// 	void IJobParallelFor.Execute(int index) {
	// 		results[index] = -1;
	// 		float3 p = queryPos[index];

	// 		int3 cell = spaceToGrid(p, oriGrid, resoGrid);

	// 		float minD = 9999.0f;
	// 		int3 curGridId;
	// 		int minRes = -1;

	// 		for (int x = -1; x <= 1; x++) {
	// 			curGridId.x = cell.x + x;
	// 			if (curGridId.x >= 0 && curGridId.x < gridDim.x) {
	// 				for (int y = -1; y <= 1; y++) {
	// 					curGridId.y = cell.y + y;
	// 					if (curGridId.y >= 0 && curGridId.y < gridDim.y) {
	// 						for (int z = -1; z <= 1; z++) {
	// 							curGridId.z = cell.z + z;
	// 							if (curGridId.z >= 0 && curGridId.z < gridDim.z) {

	// 								int neighcellhash = flatten3DTo1D(curGridId, gridDim);
	// 								int idStart = cellStartEnd[neighcellhash].x;
	// 								int idStop = cellStartEnd[neighcellhash].y;

	// 								if (idStart < int.MaxValue - 1) {
	// 									for (int id = idStart; id < idStop; id++) {

	// 										float3 posA = sortedPos[id];
	// 										float d = sqr_distance(posA, p);
	// 										if (d < minD) {
	// 											minRes = id;
	// 											minD = d;
	// 										}
	// 									}
	// 								}
	// 							}
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 		if (minRes != -1) {
	// 			results[index] = minRes;
	// 		}
	// 		else {
	// 			//Compute all the distances ! = SLOW
	// 			for (int id = 0; id < sortedPos.Length; id++) {

	// 				float3 posA = sortedPos[id];
	// 				float d = sqr_distance(p, posA);
	// 				if (d < minD) {
	// 					minRes = id;
	// 					minD = d;
	// 				}
	// 			}
	// 			results[index] = minRes;
	// 		}
	// 	}
	// 	int3 spaceToGrid(float3 pos3D, float3 originGrid, float dx) {
	// 		return (int3)((pos3D - originGrid) / dx);
	// 	}
	// 	int flatten3DTo1D(int3 id3d, int3 gridDim) {
	// 		return (gridDim.y * gridDim.z * id3d.x) + (gridDim.z * id3d.y) + id3d.z;
	// 	}

	// 	float sqr_distance(float3 p1, float3 p2) {
	// 		float x = (p1.x - p2.x) * (p1.x - p2.x);
	// 		float y = (p1.y - p2.y) * (p1.y - p2.y);
	// 		float z = (p1.z - p2.z) * (p1.z - p2.z);

	// 		return x + y + z;
	// 	}
	// }

	// [BurstCompile]
	// struct InRadiusStreamJob : IJobParallelFor {
	// 	[ReadOnly] public float3 oriGrid;
	// 	[ReadOnly] public float resoGrid;
	// 	[ReadOnly] public int3 gridDim;
	// 	[ReadOnly] public NativeArray<float3> queryPos;
	// 	[ReadOnly] public NativeArray<int2> cellStartEnd;
	// 	[ReadOnly] public NativeArray<float3> sortedPos;
	// 	[ReadOnly] public float radius;
	// 	[ReadOnly] public float radrad;
	// 	[ReadOnly] public NativeArray<int2> hashIndex;


	// 	// public NativeMultiHashMap<int, int>.Concurrent results;
	// 	public NativeStream.Writer results;

	// 	void IJobParallelFor.Execute(int index) {

	// 		results.BeginForEachIndex(index);

	// 		float3 p = queryPos[index];
	// 		int3 cell = spaceToGrid(p, oriGrid, resoGrid);

	// 		int3 curGridId;
	// 		int range = math.max(1, (int)(radius / resoGrid));

	// 		for (int x = -range; x <= range; x++) {
	// 			curGridId.x = cell.x + x;
	// 			if (curGridId.x >= 0 && curGridId.x < gridDim.x) {
	// 				for (int y = -range; y <= range; y++) {
	// 					curGridId.y = cell.y + y;
	// 					if (curGridId.y >= 0 && curGridId.y < gridDim.y) {
	// 						for (int z = -range; z <= range; z++) {
	// 							curGridId.z = cell.z + z;
	// 							if (curGridId.z >= 0 && curGridId.z < gridDim.z) {

	// 								int neighcellhash = flatten3DTo1D(curGridId, gridDim);
	// 								int idStart = cellStartEnd[neighcellhash].x;
	// 								int idStop = cellStartEnd[neighcellhash].y;

	// 								if (idStart < int.MaxValue - 1) {//Not empty cell
	// 									for (int id = idStart; id < idStop; id++) {

	// 										if (id > 0 && id < sortedPos.Length) {
	// 											float3 posA = sortedPos[id];
	// 											float d = sqr_distance(posA, p);
	// 											// Debug.Log(curGridId + " / "+id+" / "+math.sqrt(d));
	// 											if (d <= radrad) {
	// 												results.Write(hashIndex[id].y);
	// 												// results.Add(index, hashIndex[id].y);
	// 											}
	// 										}
	// 									}
	// 								}
	// 							}
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 		results.EndForEachIndex();

	// 	}
	// 	int3 spaceToGrid(float3 pos3D, float3 originGrid, float dx) {
	// 		return (int3)((pos3D - originGrid) / dx);
	// 	}
	// 	int flatten3DTo1D(int3 id3d, int3 gridDim) {
	// 		return (gridDim.y * gridDim.z * id3d.x) + (gridDim.z * id3d.y) + id3d.z;
	// 	}
	// 	float sqr_distance(float3 p1, float3 p2) {
	// 		float x = (p1.x - p2.x) * (p1.x - p2.x);
	// 		float y = (p1.y - p2.y) * (p1.y - p2.y);
	// 		float z = (p1.z - p2.z) * (p1.z - p2.z);

	// 		return x + y + z;
	// 	}
	// }

	// [BurstCompile]
	// struct InRadiusArrayJob : IJobParallelFor {
	// 	[ReadOnly] public float3 oriGrid;
	// 	[ReadOnly] public float resoGrid;
	// 	[ReadOnly] public int3 gridDim;
	// 	[ReadOnly] public NativeArray<float3> queryPos;
	// 	[ReadOnly] public NativeArray<int2> cellStartEnd;
	// 	[ReadOnly] public NativeArray<float3> sortedPos;
	// 	[ReadOnly] public float radius;
	// 	[ReadOnly] public float radrad;
	// 	[ReadOnly] public NativeArray<int2> hashIndex;
	// 	[ReadOnly] public int maxRes;

	// 	[NativeDisableParallelForRestriction]
	// 	public NativeArray<int> results;

	// 	void IJobParallelFor.Execute(int index) {

	// 		int offset = index * maxRes;
	// 		for (int i = 0; i < maxRes; i++) {
	// 			results[offset + i] = -1;
	// 		}

	// 		float3 p = queryPos[index];
	// 		int3 cell = spaceToGrid(p, oriGrid, resoGrid);

	// 		int3 curGridId;
	// 		int range = math.max(1, (int)(radius / resoGrid));

	// 		int nbRes = 0;
	// 		for (int x = -range; x <= range; x++) {
	// 			curGridId.x = cell.x + x;
	// 			if (curGridId.x >= 0 && curGridId.x < gridDim.x) {
	// 				for (int y = -range; y <= range; y++) {
	// 					curGridId.y = cell.y + y;
	// 					if (curGridId.y >= 0 && curGridId.y < gridDim.y) {
	// 						for (int z = -range; z <= range; z++) {
	// 							curGridId.z = cell.z + z;
	// 							if (curGridId.z >= 0 && curGridId.z < gridDim.z) {

	// 								int neighcellhash = flatten3DTo1D(curGridId, gridDim);
	// 								int idStart = cellStartEnd[neighcellhash].x;
	// 								int idStop = cellStartEnd[neighcellhash].y;

	// 								if (idStart < int.MaxValue - 1) {//Not empty cell
	// 									for (int id = idStart; id < idStop; id++) {

	// 										if (id > 0 && id < sortedPos.Length) {
	// 											float3 posA = sortedPos[id];
	// 											float d = sqr_distance(posA, p);
	// 											// Debug.Log(curGridId + " / "+id+" / "+math.sqrt(d));
	// 											if (d <= radrad) {
	// 												results[offset + nbRes] = hashIndex[id].y;
	// 												nbRes++;
	// 											}
	// 										}
	// 									}
	// 								}
	// 							}
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// 	int3 spaceToGrid(float3 pos3D, float3 originGrid, float dx) {
	// 		return (int3)((pos3D - originGrid) / dx);
	// 	}
	// 	int flatten3DTo1D(int3 id3d, int3 gridDim) {
	// 		return (gridDim.y * gridDim.z * id3d.x) + (gridDim.z * id3d.y) + id3d.z;
	// 	}
	// 	float sqr_distance(float3 p1, float3 p2) {
	// 		float x = (p1.x - p2.x) * (p1.x - p2.x);
	// 		float y = (p1.y - p2.y) * (p1.y - p2.y);
	// 		float z = (p1.z - p2.z) * (p1.z - p2.z);

	// 		return x + y + z;
	// 	}
	// }

	// [BurstCompile]
	// struct InRadiusMergeJob : IJobParallelFor {
	// 	[ReadOnly] public float3 oriGrid;
	// 	[ReadOnly] public float resoGrid;
	// 	[ReadOnly] public int3 gridDim;
	// 	[ReadOnly] public NativeArray<float3> queryPos;
	// 	[ReadOnly] public NativeArray<int2> cellStartEnd;
	// 	[ReadOnly] public NativeArray<float3> sortedPos;
	// 	[ReadOnly] public float radius;
	// 	[ReadOnly] public float radrad;
	// 	[ReadOnly] public NativeArray<int2> hashIndex;


	// 	public NativeQueue<int> results;

	// 	void IJobParallelFor.Execute(int index) {

	// 		float3 p = queryPos[index];
	// 		int3 cell = spaceToGrid(p, oriGrid, resoGrid);

	// 		int3 curGridId;
	// 		int range = math.max(1, (int)(radius / resoGrid));

	// 		for (int x = -range; x <= range; x++) {
	// 			curGridId.x = cell.x + x;
	// 			if (curGridId.x >= 0 && curGridId.x < gridDim.x) {
	// 				for (int y = -range; y <= range; y++) {
	// 					curGridId.y = cell.y + y;
	// 					if (curGridId.y >= 0 && curGridId.y < gridDim.y) {
	// 						for (int z = -range; z <= range; z++) {
	// 							curGridId.z = cell.z + z;
	// 							if (curGridId.z >= 0 && curGridId.z < gridDim.z) {

	// 								int neighcellhash = flatten3DTo1D(curGridId, gridDim);
	// 								int idStart = cellStartEnd[neighcellhash].x;
	// 								int idStop = cellStartEnd[neighcellhash].y;

	// 								if (idStart < int.MaxValue - 1) {//Not empty cell
	// 									for (int id = idStart; id < idStop; id++) {

	// 										float3 posA = sortedPos[id];
	// 										float d = sqr_distance(posA, p);
	// 										// Debug.Log(curGridId + " / "+id+" / "+math.sqrt(d));
	// 										if (d <= radrad) {
	// 											results.Enqueue(hashIndex[id].y);
	// 										}
	// 									}
	// 								}
	// 							}
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}

	// 	}
	// 	int3 spaceToGrid(float3 pos3D, float3 originGrid, float dx) {
	// 		return (int3)((pos3D - originGrid) / dx);
	// 	}
	// 	int flatten3DTo1D(int3 id3d, int3 gridDim) {
	// 		return (gridDim.y * gridDim.z * id3d.x) + (gridDim.z * id3d.y) + id3d.z;
	// 	}
	// 	float sqr_distance(float3 p1, float3 p2) {
	// 		float x = (p1.x - p2.x) * (p1.x - p2.x);
	// 		float y = (p1.y - p2.y) * (p1.y - p2.y);
	// 		float z = (p1.z - p2.z) * (p1.z - p2.z);

	// 		return x + y + z;
	// 	}
	// }




	public struct int2Comparer : IComparer<int2> {
		public int Compare(int2 lhs, int2 rhs) {
			return lhs.x.CompareTo(rhs.x);
		}
	}

	public struct intInvComparer : IComparer<int> {
		public int Compare(int lhs, int rhs) {
			return rhs.CompareTo(lhs);
		}
	}

	//From https://gist.github.com/LotteMakesStuff/c2f9b764b15f74d14c00ceb4214356b4
	unsafe void GetNativeArray(NativeArray<float3> posNativ, Vector3[] posArray)
	{

		// pin the buffer in place...
		fixed (void* bufferPointer = posArray)
		{
			// ...and use memcpy to copy the Vector3[] into a NativeArray<floar3> without casting. whould be fast!
			UnsafeUtility.MemCpy(NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(posNativ),
			                     bufferPointer, posArray.Length * (long) UnsafeUtility.SizeOf<float3>());
		}
		// we only have to fix the .net array in place, the NativeArray is allocated in the C++ side of the engine and
		// wont move arround unexpectedly. We have a pointer to it not a reference! thats basically what fixed does,
		// we create a scope where its 'safe' to get a pointer and directly manipulate the array

	}
	unsafe static void SetNativeArrayInt(int[] arr, NativeArray<int> nativ)
	{
		// pin the target array and get a pointer to it
		fixed (void* arrPointer = arr)
		{
			// memcopy the native array over the top
			UnsafeUtility.MemCpy(arrPointer, NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(nativ), arr.Length * (long) UnsafeUtility.SizeOf<int>());
		}
	}
}
}