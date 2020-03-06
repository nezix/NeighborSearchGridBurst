using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using System.Linq;



namespace Nezix {
public class NeighborSearchExample : MonoBehaviour {

	public int Npoints = 5000;
	public int NQueryPoints = 100;
	public float cutoff = 0.5f;
	public int maxRes = 1000;

	const int BATCH_MAX = 1023;
	const float BATCH_MAX_FLOAT = 1023f;

	MeshFilter mf;
	MeshRenderer mr;
	private Matrix4x4[] matrices;
	Matrix4x4[][] batchedMatrices;
	Material meshMaterial;

	NativeArray<float3> allPoints;
	NativeArray<float3> queryPoints;


	void Start() {

		allPoints = new NativeArray<float3>(Npoints, Allocator.Persistent);
		queryPoints = new NativeArray<float3>(NQueryPoints, Allocator.Persistent);

		for (int i = 0; i < Npoints; i++) {
			allPoints[i] = UnityEngine.Random.insideUnitSphere * 10.0f;
		}
		for (int i = 0; i < NQueryPoints; i++) {
			queryPoints[i] = UnityEngine.Random.insideUnitSphere * 10.0f;
		}


		initRendering();

		NeighborSearchGridBurst nsgb = new NeighborSearchGridBurst();

		NativeArray<int> result = nsgb.getPointsInRadius(allPoints, queryPoints, maxRes, cutoff);

		
		
		int count = 0;
		for (int i = 0; i < result.Length; i++) {
			if (result[i] == -1) {
				break;
			}
			Debug.Log(i + " : " + result[i]);
			count++;
		}
		Debug.Log("Count = " + count);

		result.Dispose();


	}

	private void initRendering() {
		mf = gameObject.AddComponent<MeshFilter>();
		// mr = gameObject.AddComponent<MeshRenderer>();
		meshMaterial = Resources.Load("Materials/InstancedStandard") as Material;

		matrices = new Matrix4x4[Npoints];
		int batches = Mathf.CeilToInt(Npoints / BATCH_MAX_FLOAT);
		batchedMatrices = new Matrix4x4[batches][];
		for (int i = 0; i < batches; i++) {
			batchedMatrices[i] = new Matrix4x4[BATCH_MAX];
		}

		GameObject test = GameObject.CreatePrimitive(PrimitiveType.Cube);
		mf.mesh = test.GetComponent<MeshFilter>().mesh;
		GameObject.Destroy(test);

		for (int i = 0 ; i < Npoints; i++) {
			matrices[i] = Matrix4x4.identity;
			matrices[i].SetTRS(
			    allPoints[i],
			    Quaternion.identity,
			    Vector3.one * 0.1f);
		}

		for (int i = 0; i < batches; ++i) {
			int batchCount = Mathf.Min(1023, Npoints - (BATCH_MAX * i));
			int start = Mathf.Max(0, (i - 1) * BATCH_MAX);

			batchedMatrices[i] = GetBatchedMatrices(start, batchCount);
		}
	}

	void Update() {

		int batches = Mathf.CeilToInt(Npoints / BATCH_MAX_FLOAT);

		for (int i = 0; i < batches; ++i) {
			int batchCount = Mathf.Min(1023, Npoints - (BATCH_MAX * i));
			Graphics.DrawMeshInstanced(mf.sharedMesh, 0, meshMaterial, batchedMatrices[i], batchCount);
		}
	}

	private Matrix4x4[] GetBatchedMatrices(int offset, int batchCount)
	{
		Matrix4x4[] ms = new Matrix4x4[BATCH_MAX];

		for (int i = 0; i < batchCount; ++i)
		{
			ms[i] = matrices[i + offset];
		}

		return ms;
	}

	void OnDestroy() {
		allPoints.Dispose();
		queryPoints.Dispose();
	}

}
}