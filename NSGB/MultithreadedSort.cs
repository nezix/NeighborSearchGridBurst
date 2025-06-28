using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace BurstGridSearch
{
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
    
    public static class ConcreteJobs
    {
        static ConcreteJobs()
        {
            new MultithreadedSort.Merge<SortEntry>().Schedule();
            new MultithreadedSort.QuicksortJob<SortEntry>().Schedule();
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
}