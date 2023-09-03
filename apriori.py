# Sebastian Raschka 2014-2022
# myxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import ipdb
from ..frequent_patterns import fpcommon as fpc
from constant import DATA_DIR, MIMIC_2_DIR, MIMIC_3_DIR, ICD_50_RANK
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
import re
global_dict_df = dict()
def generate_new_combinations(old_combinations):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: np.array
        All combinations with enough support in the last step
        Combinations are represented by a matrix.
        Number of columns is equal to the combination size
        of the previous step.
        Each row represents one combination
        and contains item type ids in the ascending order
        ```
               0        1
        0      15       20
        1      15       22
        2      17       19
        ```

    Returns
    -----------
    Generator of all combinations from the last step x items
    from the previous step.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori

    """

    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = items_types_in_previous_step > max_combination
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        for item in valid_items:
            yield from old_tuple
            yield item


def generate_new_combinations_low_memory(old_combinations, X, min_support, is_sparse):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: np.array
        All combinations with enough support in the last step
        Combinations are represented by a matrix.
        Number of columns is equal to the combination size
        of the previous step.
        Each row represents one combination
        and contains item type ids in the ascending order
        ```
               0        1
        0      15       20
        1      15       22
        2      17       19
        ```

    X: np.array or scipy sparse matrix
      The allowed values are either 0/1 or True/False.
      For example,

    ```
        0     True False  True  True False  True
        1     True False  True False False  True
        2     True False  True False False False
        3     True  True False False False False
        4    False False  True  True  True  True
        5    False False  True False  True  True
        6    False False  True False  True False
        7     True  True False False False False
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minumum support of the itemsets returned.
      The support is computed as the fraction
      `transactions_where_item(s)_occur / total_transactions`.

    is_sparse : bool True if X is sparse

    Returns
    -----------
    Generator of all combinations from the last step x items
    from the previous step. Every combination contains the
    number of transactions where this item occurs, followed
    by item type ids in the ascending order.
    No combination other than generated
    do not have a chance to get enough support

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

    """

    items_types_in_previous_step = np.unique(old_combinations.flatten())
    rows_count = X.shape[0]
    threshold = min_support * rows_count
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = items_types_in_previous_step > max_combination
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        if is_sparse:
            mask_rows = X[:, old_tuple].toarray().all(axis=1)
            X_cols = X[:, valid_items].toarray()
            supports = X_cols[mask_rows].sum(axis=0)
        else:
            mask_rows = X[:, old_tuple].all(axis=1)
            supports = X[mask_rows][:, valid_items].sum(axis=0)
        valid_indices = (supports >= threshold).nonzero()[0]
        for index in valid_indices:
            yield supports[index]
            yield from old_tuple
            yield valid_items[index]

def confidence(x, df,min_confidence):
    items = x.to_list()
    flag = 0
    n = df[items].loc[(df[items] != False).all(axis=1)].shape[0]
    for icd,_ in ICD_50_RANK:
        items.append('ICD'+icd)
        tempvalues = df[items].loc[(df[items] != False).all(axis=1)].shape[0]
        if float(tempvalues) / n >= min_confidence:
            flag = 1
            break
        items.remove('ICD'+icd)
    if flag == 0:
        return False
    return True

index_columns = dict()

def cal(X):
    return X.sum()
def cal_support(X):
    m = X.shape[1]
    return (X.sum(axis=1) == m).sum()
def convert_global(x):
    global filtered_df
    filtered_df = x
def cal1(df_word):
    return df_word.loc[(df_word!=False).all(axis=1)].shape[0]
def cal2(df_item):
    return df_item.loc[(df_item != False).all(axis=1)].shape[0]

def apriori(
    df, labels, set_labels=[], min_support=0.5, min_confidence=0.9, use_colnames=False, max_len=None, verbose=0, low_memory=False
):
    """Get frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame the encoded format. Also supports
      DataFrames with sparse data; for more info, please
      see (https://pandas.pydata.org/pandas-docs/stable/
           user_guide/sparse.html#sparse-data-structures)

      Please note that the old pandas SparseDataFrame format
      is no longer supported in mlxtend >= 0.17.2.

      The allowed values are either 0/1 or True/False.
      For example,

    ```
             Apple  Bananas   Beer  Chicken   Milk   Rice
        0     True    False   True     True  False   True
        1     True    False   True    False  False   True
        2     True    False   True    False  False  False
        3     True     True  False    False  False  False
        4    False    False   True     True   True   True
        5    False    False   True    False   True   True
        6    False    False   True    False   True  False
        7     True     True  False    False  False  False
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minumum support of the itemsets returned.
      The support is computed as the fraction
      `transactions_where_item(s)_occur / total_transactions`.

    use_colnames : bool (default: False)
      If `True`, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths (under the apriori condition) are evaluated.

    verbose : int (default: 0)
      Shows the number of iterations if >= 1 and `low_memory` is `True`. If
      >=1 and `low_memory` is `False`, shows the number of combinations.

    low_memory : bool (default: False)
      If `True`, uses an iterator to search for combinations above
      `min_support`.
      Note that while `low_memory=True` should only be used for large dataset
      if memory resources are limited, because this implementation is approx.
      3-6x slower than the default.


    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).
      Each itemset in the 'itemsets' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

    """

    def _support(_x, _n_rows, _is_sparse):
        """DRY private method to calculate support as the
        row-wise sum of values / number of rows

        Parameters
        -----------

        _x : matrix of bools or binary

        _n_rows : numeric, number of rows in _x

        _is_sparse : bool True if _x is sparse

        Returns
        -----------
        np.array, shape = (n_rows, )

        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

        """
        out = np.sum(_x, axis=0) / _n_rows
        return np.array(out).reshape(-1)

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )

    fpc.valid_input_check(df)

    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            X = df.values
        else:
            X = df.sparse.to_coo().tocsc()
        is_sparse = True
    else:
        # dense DataFrame
        X = df.values
        is_sparse = False

    support = _support(X, X.shape[0], is_sparse)
    ary_col_idx = np.arange(X.shape[1])
    flag_index = support >= min_support
    
    
    
    min_icd_confidence = dict()
    col_name = df.columns
    col_name_index = dict()
    for i in range(df.shape[1]):
        col_name_index[col_name[i]] = i


    for i in range(col_name.shape[0]):
        index_columns[col_name[i]] = i
    params = []
    tf = Pool(cpu_count())
    for i in range(X.shape[1]):
        params.append(X[:, i])
    df_word_nums = tf.map(cal, params)
    
    list_icd50_df = []
    icd50 = set_labels
    for icd in icd50:
        index_X = np.where(X[:, col_name_index[icd]]==True)[0]
        list_icd50_df.append(X[index_X, :])

    params = []
    dict_params_name = []
    num_params = 0
    delete_index = []
    for i in range(len(icd50)):
        temp_X = list_icd50_df[i]
        for j in range(X.shape[1]):
            dict_params_name.append(j)
            if col_name[j][:3] == 'ICD':
                delete_index.append(num_params)
            params.append(temp_X[:, j])
            num_params += 1
    rules_items = tf.map(cal, params)
    del params
    rules_items = np.array(rules_items)
    n_items = X.shape[1]
    n_icd = len(icd50)
    support_items = np.array(df_word_nums)
    confidence_rules_items = np.zeros(rules_items.shape[0], float)
    for i in range(n_icd):
        confidence_rules_items[i*n_items: (i+1)*n_items] = np.divide(rules_items[i*n_items: (i+1)*n_items], support_items)
    flag_index = confidence_rules_items >= min_confidence
    flag_index[np.array(delete_index)] = False
    flag_index = np.where(flag_index == True)[0]
    dict_params_name = np.array(dict_params_name)
    support_index = np.where(support >= min_support)[0]
    selected_index = list(set(support_index).intersection(set(dict_params_name[flag_index])))

    selected_index = set(selected_index)
    set_delete_index = set(delete_index)
    for i in range(len(icd50)):
        temp_X = list_icd50_df[i]
        sum_temp_X = temp_X[:, list(selected_index)].sum(axis=1)
        index = np.where(sum_temp_X==0)[0]
        for j in index:
            if temp_X[j, list(selected_index)].sum() !=0:
                continue
            index_word = np.where(temp_X[j, :] == True)[0] + i*n_items
            index_delete_word = []
            for index_delete in index_word:
                if dict_params_name[index_delete] not in set_delete_index:
                    index_delete_word.append(index_delete)
            temp_index = list(set(dict_params_name[index_word]) - set_delete_index)
            selected_index.add(temp_index[np.argmax(confidence_rules_items[index_delete_word])])
    selected_index = list(selected_index)



    selected_num = len(selected_index)
    num_filtered = ary_col_idx[selected_index].shape[0]
    support_dict = {1: support[selected_index]}
    itemset_dict = {1: ary_col_idx[selected_index].reshape(-1, 1)}

    max_itemset = 1
    rows_count = float(X.shape[0])


    all_ones = np.ones((int(rows_count), 1))
    filtered_df = df[df.columns[selected_index].to_list()+icd50]
    n_icd = len(icd50)
    while max_itemset and max_itemset < (max_len or float("inf")):
        next_max_itemset = max_itemset + 1

        # With exceptionally large datasets, the matrix operations can use a
        # substantial amount of memory. For low memory applications or large
        # datasets, set `low_memory=True` to use a slower but more memory-
        # efficient implementation.
        if low_memory:
            combin = generate_new_combinations_low_memory(
                itemset_dict[max_itemset], X, min_support, is_sparse
            )
            # slightly faster than creating an array from a list of tuples
            combin = np.fromiter(combin, dtype=int)
            combin = combin.reshape(-1, next_max_itemset + 1)
            if combin.size == 0:
                break
            if verbose:
                print(
                    "\rProcessing %d combinations | Sampling itemset size %d"
                    % (combin.size, next_max_itemset),
                    end="",
                )
            
            flag_index = np.zeros(combin[:, 1:].shape[0], bool)

            params = []
            for i in range(combin[:, 1:].shape[0]):
                items = combin[i, 1:].tolist()
                params.append(X[:, items])
            support_items = tf.map(cal_support, params)


            params = []
            dict_params_name = []
            for i in range(len(icd50)):
                icd = icd50[i]
                temp_X = list_icd50_df[i]
                for j in range(combin[:, 1:].shape[0]):
                    items = combin[j, 1:].tolist()
                    dict_params_name.append(j)
                    params.append(temp_X[:, items])
            rules_items = tf.map(cal_support, params)
            rules_items = np.array(rules_items)
            n_items = combin[:, 1:].shape[0]
            n_icd = len(icd50)
            support_items = np.array(support_items)
            confidence_rules_items = np.zeros(rules_items.shape[0], float)
            max_confidence = np.zeros((combin.shape[0],1), float)
            for i in range(n_icd):
                max_confidence = np.max(np.concatenate([max_confidence, confidence_rules_items[i*n_items: (i+1)*n_items].reshape(-1, 1)], axis=1), axis=1).reshape(-1, 1)
                confidence_rules_items[i*n_items: (i+1)*n_items] = np.divide(rules_items[i*n_items: (i+1)*n_items], support_items)
            
            flag_index = np.where(confidence_rules_items >= min_confidence-0.15)[0]
            dict_params_name = np.array(dict_params_name)
            selected_index = np.unique(dict_params_name[flag_index])
            #combin = np.concatenate([combin, max_confidence], axis=1)
            combin = combin[selected_index,:]
            """
            support_index = np.argsort(-combin[:, 0])
            support_index = np.where(support_index <= int(selected_num/2))[0]
            confidence_index = np.argsort(-combin[:, -1])
            confidence_index = np.where(confidence_index <= int(selected_num/2))[0]
            selected_index = np.unique(np.concatenate([support_index, confidence_index]))
            combin = combin[selected_index, :]
            """
            
            itemset_dict[next_max_itemset] = combin[:, 1:]
            support_dict[next_max_itemset] = combin[:, 0].astype(float) / rows_count
            max_itemset = next_max_itemset
        else:
            combin = generate_new_combinations(itemset_dict[max_itemset])
            combin = np.fromiter(combin, dtype=int)
            combin = combin.reshape(-1, next_max_itemset)

            if combin.size == 0:
                break
            if verbose:
                print(
                    "\rProcessing %d combinations | Sampling itemset size %d"
                    % (combin.size, next_max_itemset),
                    end="",
                )

            if is_sparse:
                _bools = X[:, combin[:, 0]] == all_ones
                for n in range(1, combin.shape[1]):
                    _bools = _bools & (X[:, combin[:, n]] == all_ones)
            else:
                _bools = np.all(X[:, combin], axis=2)

            support = _support(np.array(_bools), rows_count, is_sparse)
            _mask = (support >= min_support).reshape(-1)
            if any(_mask):
                itemset_dict[next_max_itemset] = np.array(combin[_mask])
                support_dict[next_max_itemset] = np.array(support[_mask])
                max_itemset = next_max_itemset
            else:
                # Exit condition
                break

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ["support", "itemsets"]
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used

    return res_df
