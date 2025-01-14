'Loader for data from UKBB.'
import os
import glob

import numpy as np
import pandas as pd


def list_files(imaging_folder, file_pattern, assertion=True):
    'List sorted files in folder following given regex pattern.'
    file_list = glob.iglob(
        os.path.abspath(os.path.join(imaging_folder, '**'+ file_pattern)),
        recursive=True)

    file_list = list(sorted(file_list))
    # Create DataFrame with Ids and Values
    _ids = np.asarray(
        [os.path.split(f)[-1] for f in file_list]).astype(str)
    file_df = pd.DataFrame(
        np.asarray([_ids, np.asarray(file_list)]).T,
        columns=['ids', 'values']
    )
    if not assertion:
        return file_list, file_df

    assert len(file_list) > 0, '''
        Error! No files found with pattern "{}" 
        in folder {}.'''.format(file_pattern, os.path.abspath(imaging_folder))

    return file_list, file_df

def load_ukbb_data(
        imaging_folder, annot_folder, data_folder, sub_dataset='',
        target='age', view='', subcat='', balance_classes=False,
        num_classes=1, masked=False, attributes=True, file_format='.nii'):
    '''
    List files for UKBB according to given parameters.
    Parameters:
        imaging_folder [str]: path to folder with UKBB imaging data.
        annot_folder [str]: path to folder with UKBB imaging data annotations.
        data_folder [str]: path to folder with tabular files from UKBB.
        sub_dataset [str]: path to file with subsample of subjects to consider.
        target [str]: label used as target output.
        view [str]: short-axis (sa) or long-axis (la).
        subcat [str]: subcategory of view, if exist. Only used 
        for LA images. Must be one of 2ch, 3ch, 4ch.
        balance_classes [bool]: Whether or not to apply class balancing.
        num_classes [int]: Number of classes to consider.
        masked [bool]: whether or not to mask the final dataset.
        attributes [bool]: whether or not to get attributes/data from subjects.
        file_format [str]: ending of the filenames.
    Return:
        A list of filepaths and a dict with extra information (ids, 
        age, weigth, etc.).
    '''
    # Pattern to match with glob for type of image of interest.
    str_pattern = view
    if len(subcat) > 0:
        str_pattern += '_{}'.format(subcat)
    str_pattern += file_format
    file_list, file_df = list_files(imaging_folder, str_pattern)

    ##########
    # str_pattern = 'label_' + str_pattern
    # _, annot_df = list_files(annot_folder, str_pattern, False)
    ##########
    'Annotation loading (Manual)'
    annot_folder = '/NFS/FutureBrainGen/data/long/long_output.csv'
    annot_df = pd.read_csv(
        annot_folder,
        low_memory=False, dtype={0: 'str'})

    _ids = np.asarray(
        [os.path.split(f)[-1] for f in file_list]).astype(str)
    annot_df = pd.DataFrame(
        np.asarray([_ids, np.asarray(file_list)]).T,
        columns=['ids', 'values']
    )
    print('annot_df : ',annot_df)
    #########
    
    # Dataframe for handling mask of annotations (it must be same length as 'list')
    file_df = file_df.merge(annot_df, how='left', on='ids', suffixes=('_file', '_annot'))

    # Avoid repetition
    files = {
        'list': np.asarray(file_list),
        'annot': {'values': file_df.values_annot.values}}
    files['ids'] = np.asarray(
        [os.path.split(os.path.dirname(f))[-1] for f in files['list']]).astype(str)
    # Get annotation ids and compute mask for them
    files['annot']['mask'] = ~pd.isnull(file_df.values_annot)
    print("load_ukbb 'files' : ", files)

    if attributes:
        get_attributes(data_folder, files)

    # Filter data by non-null values
    mk = files[target]['mask']
    if attributes:
        mk = mk & files['MMSE_B']['mask'] ## NEED TO CHANGE TO OUR ATTRIBUTES
    if not masked:
        mk = np.ones(len(mk)) == 1
    # Filter Dataframe also by ids with attributes
    file_df = file_df[file_df.ids.isin(files['ids'])]
    # Filter out by subsample (if provided)
    if sub_dataset != '':
        # Dataset should be a list of IDs
        subset = pd.read_csv(sub_dataset, dtype=str).ID.values
        # Intersection of subjects with given label and listed in dataset file.
        mk = mk & file_df.ids.isin(subset).values

    # Balance classes if passed as settings
    if balance_classes:
        # Restrict mask even more
        _values = files[target]['values']
        _classes, _counts = np.unique(_values[mk], return_counts=True)
        if num_classes > 1:
            _classes = _classes[:num_classes]
            _counts = _counts[:num_classes]
        else: # num_classes is 1 and outputs are binary (0 and 1)
            _classes = _classes[:2]
            _counts = _counts[:2]
        _min_count = np.min(_counts)
        # Filter by classes in num_classes
        mk = mk & pd.Series(_values).isin(_classes).values
        # Select same amount of subjects for every class
        for cl in _classes:
            aux = np.where((_values == cl)&(mk == True))[0]
            false_ind = aux[_min_count:]
            # Drop subjects after a common minimum number
            if false_ind.size > 0:
                mk[false_ind] = False

    file_list = files['list'][mk]
    labels = files[target]['values'][mk]
    if 'data_array' in files.keys():
        file_data = files['data_array'][mk]
    else:
        file_data = file_list

    print('load_ukbb output:',file_list[:5], labels[:5], file_data[:5])

    return file_list, labels, file_data


def get_attributes(data_folder, file_dict):
    '''
    Obtain attributes/labels for files in input dictionary and modify it.
    Extra keys are age, weight, bmi, height, etc.
    A mask is provided for each field, so that one can filter the ids for
    which there are non-null values.
    Parameters:
        data_folder [str]: path to folder with files from UKBB.
        file_dict [dict]: dictionary with keys 'list' and 'ids'
        containing list of files and their ids, respectively.
    Return:
        True when finished.
    '''
    ukbb_df = pd.read_csv(
        os.path.join(data_folder, 'long_output.csv'),
        low_memory=False, dtype={0: 'str'})

    ukbb_df.sort_values(by='SubID', inplace=True)

    # Common user IDs
    users = set(ukbb_df.SubID.values).intersection(file_dict['ids'])
    # Filter DF by imaging available
    ukbb_df = ukbb_df[ukbb_df.SubID.isin(users)]

    # Filter IDs by available information in csv
    _mask = pd.Series(file_dict['ids']).isin(users)
    file_dict['list'] = file_dict['list'][_mask]
    file_dict['ids'] = file_dict['ids'][_mask]
    # Filter mask for annotations according to info available at ukbb_df
    file_dict['annot']['mask'] = file_dict['annot']['mask'][_mask]
    file_dict['annot']['values'] = file_dict['annot']['values'][_mask]

    file_dict['sex'] = {
        'values': ukbb_df.Sex.values,
        'mask': ~pd.isnull(ukbb_df.Sex.values)}
    file_dict['age'] = {
        'values': ukbb_df.Age_B.values,
        'mask': ~pd.isnull(ukbb_df.Age_B.values)}
    file_dict['age_F'] = {
        'values': ukbb_df.Age_F.values,
        'mask': ~pd.isnull(ukbb_df.Age_F.values)}
    file_dict['MMSE_B'] = {
        'values': ukbb_df.MMSE_B.values,
        'mask': ~pd.isnull(ukbb_df.MMSE_B.values)}
    file_dict['CDR_B'] = {
        'values': ukbb_df.CDR_B.values,
        'mask': ~pd.isnull(ukbb_df.CDR_B.values)}
    file_dict['Group_B'] = {
        'values': ukbb_df.Group_B.values,
        'mask': ~pd.isnull(ukbb_df.Group_B.values)}
    file_dict['Education'] = {
        'values': ukbb_df.Education.values,
        'mask': ~pd.isnull(ukbb_df.Education.values)}
    file_dict['Interval'] = {
        'values': ukbb_df.Interval.values,
        'mask': ~pd.isnull(ukbb_df.Interval.values)}

    # Array with data to pass together with images
    file_dict['data_array'] = np.vstack((
        file_dict['ids'], file_dict['sex']['values'],
        file_dict['age']['values'], file_dict['age_F']['values'],
        file_dict['MMSE_B']['values'],
        file_dict['CDR_B']['values'],
        file_dict['Group_B']['values'],
        file_dict['Education']['values'], file_dict['Interval']['values']
    )).astype(float).T

    return True


if __name__ == '__main__':
    # Some examples
    imaging_folder = ''
    data_folder = ''
    print(load_ukbb_data(imaging_folder, '', data_folder, None, 'age'))

    # Example of filtering by subjects with non-null BMI
    _list, _dict = load_ukbb_data(imaging_folder, data_folder, None, 'age')
    print('=> non-null values', _dict['ids'][_dict['MMSE_B']['mask']])
    print('=> null values', _dict['ids'][~_dict['CDR_B']['mask']])
