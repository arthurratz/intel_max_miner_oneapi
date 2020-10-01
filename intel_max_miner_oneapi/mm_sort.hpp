#pragma once

template<class Container, class _Pred>
void qsort3w(Container& array, std::size_t _First, std::size_t _Last, _Pred compare)
{
	if (_First >= _Last) return;

	//std::size_t _Size = _Last - _First + 1;
	//if (_Size > 0)
	{
		std::size_t _Left = _First, _Right = _Last - 1;
		bool is_swapped_left = false, is_swapped_right = false;
		MMN_RULE _Pivot = array[_First];

		std::size_t _Fwd = _First + 1;
		while (_Fwd <= _Right)
		{
			if (compare(array[_Fwd], _Pivot))
			{
				is_swapped_left = true;
				std::swap(array[_Left], array[_Fwd]);
				_Left++; _Fwd++;
			}

			else if (compare(_Pivot, array[_Fwd])) {
				is_swapped_right = true;
				std::swap(array[_Right], array[_Fwd]);
				_Right--;
			}

			else _Fwd++;
		}

		tbb::task_group task_group;
		task_group.run([&]() {
			if (((_Left - _First) > 0) && (is_swapped_left))
				qsort3w(array, _First, _Left - 1, compare);
			});

		task_group.run([&]() {
			if (((_Last - _Right) > 0) && (is_swapped_right))
				qsort3w(array, _Right + 1, _Last, compare);
			});

		task_group.wait();
	}
}

template<class Container, class _Pred >
void parallel_sort(Container& array, std::size_t _First, \
	std::size_t _Last, _Pred compare)
{
	tbb::task_group task_group;
	if ((_Last - _First) > 1)
	{
		std::size_t _Left = _First, _Right = _Last + 1;
		std::size_t _Mid = _First + (_Last - _First) / 2;
		MMN_RULE _Pivot = array[_Mid];

		while (_Left <= _Right)
		{
			while (compare(array[_Left], _Pivot)) _Left++;
			while (compare(_Pivot, array[_Right])) _Right--;

			if (_Left <= _Right) {
				std::swap(array[_Left], array[_Right]);
				_Left++; _Right--;
			}
		}

		task_group.run([&]() {
			qsort3w(array, _First, _Right, compare);
		});

		task_group.run([&]() {
			qsort3w(array, _Left, _Last - 1, compare);
		});

		task_group.wait();
	}

	else {
		task_group.run_and_wait([&]() {
			qsort3w(array, _First, _Last - 1, compare);
		});
	}
}