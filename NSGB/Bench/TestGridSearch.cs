using UnityEngine;
using UnityEngine.UI;
using System.Text;
using System.IO;
using System.Collections;

namespace BurstGridSearch.Benchmark {

public class TestGridSearch : MonoBehaviour {
	public bool testClosestPoint = true;

	public int N = 100000;
	public int K = 100000;
	public int maxNei = 50;
	public float radSearch = 2.0f;

	Vector3[] pos = null;

	public Text mytext;

	IEnumerator Start() {

		pos = new Vector3[N];

		Random.InitState(1235);
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
		int timesTest = 50;
		int[] lastresults = null;
	
		for (int i = 0; i < timesTest; i++) {

			float start = Time.realtimeSinceStartup;

			GridSearchBurst gsb = new GridSearchBurst(-1.0f, 28);

			gsb.initGrid(pos);

			float res1 = (1000.0f * (Time.realtimeSinceStartup - start));
			if (i != 0)//warmup
				meanCreation += res1;
			// UnityEngine.Debug.Log("Time for grid creation: " + (1000.0f * (Time.realtimeSinceStartup - start)).ToString("f3") + " ms");
			start = Time.realtimeSinceStartup;

			int[] results = null;
			if(testClosestPoint)
				results = gsb.searchClosestPoint(queries);
			else
		    	results = gsb.searchWithin(queries, radSearch, maxNei);
			lastresults = results;

			// UnityEngine.Debug.Log("Time for grid search: " + (1000.0f * (Time.realtimeSinceStartup - start)).ToString("f3") + " ms");
			float res2 = (1000.0f * (Time.realtimeSinceStartup - start));
			if (i != 0)//warmup
				meanQuery += res2;

			// int countWrong = 0;
			// for (int k = 0; k < results.Length; k++)
			// 	if (results[k] == -1)
			// 		countWrong++;
			// Debug.Log("Wrong = " + countWrong +" / "+queries.Length);

			// int countFull = K;
			// for (int k = 0; k < K; k++) {
			// 	for (int a = 0; a < maxNei; a++) {
			// 		if (results[k * maxNei + a] == -1) {
			// 			countFull--;
			// 			break;
			// 		}
			// 	}
			// }
			// Debug.Log("Full = " + countFull + " / " + K);

			if (mytext != null) {
				mytext.text = "Creation " + res1.ToString("f3") + "ms\nSearch = " + res2.ToString("f3") + " ms";
			}
			gsb.clean();

			// yield return new WaitForSeconds(1);
			// yield break;

		}

		meanCreation /= timesTest - 1;
		meanQuery /= timesTest - 1;

		if (mytext != null) {
			mytext.text = "Creation " + meanCreation.ToString("f3") + "ms\nSearch = " + meanQuery.ToString("f3") + " ms";
		}
		Debug.Log("Creation " + meanCreation.ToString("f3") + "ms");
		Debug.Log("Queries " + meanQuery.ToString("f3") + "ms");

		yield break;
		//Verif-------------
		// int[] trueResults = new int[queries.Length];
		// for (int i = 0; i < queries.Length; i++) {
		// 	int curClose = -1;
		// 	float minD = float.MaxValue;
		// 	for (int k = 0; k < N; k++) {
		// 		float d = Vector3.Distance(queries[i], pos[k]);
		// 		if (d < minD) {
		// 			minD = d;
		// 			curClose = k;
		// 		}
		// 	}
		// 	trueResults[i] = curClose;
		// 	if (trueResults[i] == 0) {
		// 		Debug.Log("Closest point to " + queries[i] + "  =  " + pos[trueResults[i]] + " dist = " + minD);
		// 		Debug.Log("Burst gives us " + pos[results[i]] + "  dist = " + Vector3.Distance(queries[i], pos[results[i]]) + "\n---------------");
		// 	}
		// }

		// for (int i = 0; i < queries.Length; i++) {
		// 	if (results[i] != trueResults[i]) {
		// 		Debug.LogError("Pb for result " + i + " :     " + results[i] + "         != " + trueResults[i]);
		// 		Debug.LogError(Vector3.Distance(pos[results[i]], queries[i]) + " vs " + Vector3.Distance(pos[trueResults[i]], queries[i]));
		// 	}
		// }

		//Verif within -----------------
		// int[] trueResults = new int[queries.Length * K];
		// for (int i = 0; i < trueResults.Length; i++) {
		// 	trueResults[i] = -1;
		// }
		// for (int i = 0; i < queries.Length; i++) {
		// 	int idRes = 0;
		// 	for (int k = 0; k < N; k++) {
		// 		float d = Vector3.Distance(queries[i], pos[k]);
		// 		if (d <= radSearch) {
		// 			trueResults[i * maxNei + idRes] = k;
		// 			idRes++;
		// 			if (idRes == maxNei) {
		// 				break;
		// 			}
		// 		}
		// 	}
		// }

		// for (int i = 0; i < queries.Length; i++) {
		// 	int counttrueResults = 0;
		// 	int countGridRes = 0;

		// 	for (int j = 0; j < K; j++) {
		// 		if (lastresults[i * K + j] != -1)
		// 			countGridRes++;
		// 	}

		// 	for (int j = 0; j < K; j++) {
		// 		if (trueResults[i * K + j] != -1)
		// 			counttrueResults++;
		// 	}

		// 	if (counttrueResults != countGridRes) {
		// 		Debug.LogError("Pb for query " + i + " : real count = " + counttrueResults + " |  grid count = " + countGridRes);
		// 	}
		// }

		// yield break;
	}


}
}