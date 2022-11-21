/*---------Install KNN package first
using UnityEngine;
using UnityEngine.UI;
using System.Text;
using System.IO;
using System.Collections;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

using KNN;
using KNN.Jobs;

using Random = UnityEngine.Random;

namespace BurstGridSearch.Benchmark {

public class TestKNNSearch : MonoBehaviour {


	public bool testClosestPoint = true;

	public int N = 100000;
	public int K = 100000;
	public int maxNei = 50;
	public float radSearch = 2.0f;

	Vector3[] pos = null;

	public Text mytext;

	IEnumerator Start() {

		pos = new Vector3[N];

		Random.seed = 1235;
		for (int i = 0; i < N; i++) {
			pos[i] = new Vector3(Random.Range(-100.0f, 100.0f),
			                     Random.Range(-100.0f, 100.0f),
			                     Random.Range(-100.0f, 100.0f));
		}


		Vector3[] queries = new Vector3[K];

		for (int i = 0; i < K; i++) {
			queries[i] = new Vector3(Random.Range(-100.0f, 100.0f),
			                         Random.Range(-100.0f, 100.0f),
			                         Random.Range(-100.0f, 100.0f));
		}

		float meanCreation = 0.0f;
		float meanQuery = 0.0f;
		int timesTest = 20;
		int[] lastresults = null;

		for (int i = 0; i < timesTest; i++) {

			float start = Time.realtimeSinceStartup;

			NativeArray<float3> positions = new NativeArray<float3>(pos.Length, Allocator.Persistent);
			GetNativeArray(positions, pos);

			//Create KDTree
			KnnContainer knnContainer = new KnnContainer(positions, false, Allocator.TempJob);
			NativeArray<float3> queryPositions = new NativeArray<float3>(queries.Length, Allocator.TempJob);

			GetNativeArray(queryPositions, queries);


			KnnRebuildJob rebuildJob = new KnnRebuildJob(knnContainer);
			rebuildJob.Schedule().Complete();

			float res1 = (1000.0f * (Time.realtimeSinceStartup - start));

			if (i != 0)//warmup
				meanCreation += res1;

			start = Time.realtimeSinceStartup;

			NativeArray<int> results;
			if (testClosestPoint)
				results = new NativeArray<int>(positions.Length * 1, Allocator.TempJob);
			else
				results = new NativeArray<int>(positions.Length * maxNei, Allocator.TempJob);

			var batchQueryJob = new QueryKNearestBatchJob(knnContainer, queryPositions, results);
			batchQueryJob.ScheduleBatch(queryPositions.Length, queryPositions.Length / 32).Complete();


			float res2 = (1000.0f * (Time.realtimeSinceStartup - start));
			if (i != 0)//warmup
				meanQuery += res2;

			if (mytext != null) {
				mytext.text = "Creation " + res1.ToString("f3") + "ms\nSearch = " + res2.ToString("f3") + " ms";
			}

			// yield return new WaitForSeconds(1);
		}

		meanCreation /= timesTest - 1;
		meanQuery /= timesTest - 1;

		if (mytext != null) {
			mytext.text = "Creation " + meanCreation.ToString("f3") + "ms\nSearch = " + meanQuery.ToString("f3") + " ms";
		}
		Debug.Log("Creation " + meanCreation.ToString("f3") + "ms");
		Debug.Log("Queries " + meanQuery.ToString("f3") + "ms");

		yield break;

	}
	//From https://gist.github.com/LotteMakesStuff/c2f9b764b15f74d14c00ceb4214356b4
	static unsafe void GetNativeArray(NativeArray<float3> posNativ, Vector3[] posArray)
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

}
}
*/
