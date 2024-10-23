from typing import Dict, List

import pandas as pd

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)):
            group.append(lst[i + j])
        result.extend(group[::-1])  
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    length_dict = {}
    for word in lst:
        length = len(word)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(word)
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    def _flatten(d: Dict, parent_key: str = '') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(_flatten({f"{new_key}[{i}]": item}).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    return _flatten(nested_dict)

from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
	def backtrack(start: int):
        if start == len(nums):
            results.append(nums[:])  
            return
        seen = set()  
        for i in range(start, len(nums)):
            if nums[i] not in seen:  
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  
                backtrack(start + 1)  
                nums[start], nums[i] = nums[i], nums[start]  

    results = []
    nums.sort()  
    backtrack(0)
    return results



import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    
    pattern = r'(?:(\d{2}-\d{2}-\d{4})|(\d{2}/\d{2}/\d{4})|(\d{4}\.\d{2}\.\d{2}))'
    
    matches = re.findall(pattern, text)
    
    valid_dates = [date for match in matches for date in match if date]
    
    return valid_dates

import pandas as pd
import polyline
from haversine import haversine

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    df['distance'] = 0
    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine(df.loc[i-1, ['latitude', 'longitude']], df.loc[i, ['latitude', 'longitude']])
    
    return df
from typing import List

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:

    n = len(matrix)
    
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    transformed_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            transformed_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]

    return transformed_matrix

import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:

    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    return df.groupby(['id', 'id_2']).apply(lambda g: (g['end'].max() - g['start'].min() >= pd.Timedelta(days=7)) and (g['start'].dt.date.nunique() == 7) ).rename_axis(['id', 'id_2']).reset_index(drop=True)

