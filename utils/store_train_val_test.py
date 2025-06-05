import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import random

### function that computes the mean values of each band for each sample
### note that the function computes the mean of the sample bands' means, 
### as a more parallelizable approach
### we consider it a good approximation anyway
def calculate_normalization_params(train_files, num_workers, use_indices=0, use_qai=0, indices_bands=[]):
    means = []
    stds = []

    def process_file(file, use_indices=0, use_qai=0, indices_bands=[]):
        df = pd.read_csv(file)
        ### drop the "_mean" in the column names
        df.rename(columns=lambda x: x.replace('_mean', ''), inplace=True)
        ### drop the rows with doy == 0 (if any)
        df = df.loc[df['DOY'] != 0]
        ### do not consider the doy nor the date column
        df = df.drop(columns=['DOY', 'date'])

        # ### brightness filter as in data_loader
        # df = df[(df['BLU_mean'] + df['GRN_mean'] + df['RED_mean']) < 5000]

        if use_qai == 1:
            ### convert QAI column (integer) to 15-bit binary
            df['QAI_binary'] = df['QAI'].apply(lambda x: format(x, '015b'))
            ### convert the QAI_binary column to string
            df['QAI_binary'] = df['QAI_binary'].astype(str)
            ### decipher the binary values according to the following scheme: 
            ### https://force-eo.readthedocs.io/en/latest/howto/qai.html
            ### separate this 15-bit binary number as follows
            ### from right to left: 
            ### first bit should be stored in the field "valid"
            ### second and third bit together stored in "cloud"
            ### fourth bit stored in cloud_shadow
            ### fifth bit stored in snow
            ### sixth bit stored in water
            ### seventh and eigth bit together stored in aerosol
            ### ninth bit stored in subzero
            ### tenth bit stored in saturation
            ### eleventh bit stored in high_sun_zenith
            ### twelfth and thirteenth bit together stored in illumination
            ### fourteenth bit stored in slope
            ### fifteenth bit stored in water_vapor
            df['valid'] = df['QAI_binary'].apply(lambda x: x[14])
            df['cloud'] = df['QAI_binary'].apply(lambda x: x[12:14])
            df['cloud_shadow'] = df['QAI_binary'].apply(lambda x: x[11])
            df['snow'] = df['QAI_binary'].apply(lambda x: x[10])
            df['water'] = df['QAI_binary'].apply(lambda x: x[9])
            df['aerosol'] = df['QAI_binary'].apply(lambda x: x[7:9])
            df['subzero'] = df['QAI_binary'].apply(lambda x: x[6])
            df['saturation'] = df['QAI_binary'].apply(lambda x: x[5])
            df['high_sun_zenith'] = df['QAI_binary'].apply(lambda x: x[4])
            df['illumination'] = df['QAI_binary'].apply(lambda x: x[2:4])
            df['slope'] = df['QAI_binary'].apply(lambda x: x[1])
            df['water_vapor'] = df['QAI_binary'].apply(lambda x: x[0])
            ### remove rows with the following features: 
            ### valid == 1, cloud == 10 or 11, cloud_shadow == 1, snow == 1, water == 1, aerosol == 10 or 11, 
            ### subzero == 1, saturation == 1, high_sun_zenith == 1, illumination == 10 or 11, water_vapor == 1
            ### the water_vapor decision is disputable because I'm not sure if the quality is impaired much
            ### by the presence of water vapor in the atmosphere
            df = df[(df['valid'] == "0") & (df['cloud'] != "10") & (df['cloud'] != "11") & 
                    (df['cloud_shadow'] == "0") & (df['snow'] == "0") & (df['water'] == "0") & 
                    (df['aerosol'] != "10") & (df['aerosol'] != "11") & (df['subzero'] == "0") & 
                    (df['saturation'] == "0") & (df['high_sun_zenith'] == "0") & 
                    (df['illumination'] != "10") & (df['illumination'] != "11") & 
                    (df['water_vapor'] == "0")]
            ### drop all QAI-related columns
            df = df.drop(columns=['QAI', 'QAI_binary', 'valid', 'cloud', 'cloud_shadow', 'snow', 'water', 'aerosol', 'subzero', 'saturation', 'high_sun_zenith', 'illumination', 'slope', 'water_vapor'])
        ### else means: if use_qai == 0
        else:
            ### if the QAI column exists, drop it
            if 'QAI' in df.columns:
                df = df.drop(columns=['QAI'])
        
        ### if indices_bands contains one of the indices from the indices list, 
        ### we compute the vegetation indices, and then drop all but the indices_bands columns
        # List of indices to check
        indices_to_check = ["DSWI", "NDWI", "CLRE", "NDREI2", "NDREI1", "SWIRI", "CRSWIR", "NGRDI", "SRSWIR", "LWCI"]
        # Check if at least one of the indices is in indices_bands
        if len(indices_bands) > 0:
            if any(index in indices_bands for index in indices_to_check):
                # Compute the indices
                if "DSWI" in indices_bands:
                    df['DSWI'] = (df['BNR'] + df['GRN']) / (df['SW1'] + df['RED'])
                    df = df[(df['DSWI'] >= -1) & (df['DSWI'] <= 5)]
                if "NDWI" in indices_bands:
                    df['NDWI'] = (df['GRN'] - df['BNR']) / (df['GRN'] + df['BNR'])
                    df = df[(df['NDWI'] >= -1) & (df['NDWI'] <= 1)]
                if "CLRE" in indices_bands:
                    df['CLRE'] = (df['RE3'] / df['RE1']) - 1
                    df = df[(df['CLRE'] >= -1) & (df['CLRE'] <= 10)]
                if "NDREI2" in indices_bands:
                    df['NDREI2'] = (df['RE3'] - df['RE1']) / (df['RE3'] + df['RE1'])
                    df = df[(df['NDREI2'] >= -1) & (df['NDREI2'] <= 1)]
                if "NDREI1" in indices_bands:
                    df['NDREI1'] = (df['RE2'] - df['RE1']) / (df['RE2'] + df['RE1'])
                    df = df[(df['NDREI1'] >= -1) & (df['NDREI1'] <= 1)]
                if "SWIRI" in indices_bands:
                    df['SWIRI'] = df['SW1'] / df['BNR']
                    df = df[(df['SWIRI'] >= 0) & (df['SWIRI'] <= 1)] # exception: SWIRI must be positive
                if "CRSWIR" in indices_bands:
                    df['CRSWIR'] = df['SW1'] / (df['NIR'] + ((df['SW2'] - df['NIR']) / (2185.7 - 864)) * (1610.4 - 864))
                    df = df[(df['CRSWIR'] >= -1) & (df['CRSWIR'] <= 1)]
                if "NGRDI" in indices_bands:
                    df['NGRDI'] = (df['GRN'] - df['RED']) / (df['GRN'] + df['RED'])
                    df = df[(df['NGRDI'] >= -1) & (df['NGRDI'] <= 1)]
                if "SRSWIR" in indices_bands:
                    df['SRSWIR'] = df['SW1'] / df['SW2']
                    df = df[(df['SRSWIR'] >= 0) & (df['SRSWIR'] <= 2)] ### upper cap 1 in healthy forest, we allow 2
                if "LWCI" in indices_bands:
                    df['LWCI'] = np.log(1 - (df['BNR'] - df['SW1'])) / -np.log(1 - (df['BNR'] - df['SW1']))
                    # df = df[(df['LWCI'] >= 0) & (df['LWCI'] <= 1)]
            ### drop all but the indices_bands columns and keep the order
            df = df[indices_bands]
            ### drop DataFrame rows which contain unreasonable values
            df.dropna(inplace=True)
            df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
        elif use_indices == 1:
            ### compute the vegetation indices as in dataloader and drop the band columns
            df['DSWI'] = (df['BNR'] + df['GRN']) / (df['SW1'] + df['RED'])
            df['NDWI'] = (df['GRN'] - df['BNR']) / (df['GRN'] + df['BNR'])
            # df['TCW'] = 0.1763*df['BLU'] + 0.1615*df['GRN'] + 0.0486*df['RED'] - 0.0755*df['BNR'] - 0.7701*df['SW1'] - 0.5293*df['SW2']
            df['CLRE'] = (df['RE3'] / df['RE1']) - 1
            df['NDREI2'] = (df['RE3'] - df['RE1']) / (df['RE3'] + df['RE1'])
            df['NDREI1'] = (df['RE2'] - df['RE1']) / (df['RE2'] + df['RE1'])
            df['SWIRI'] = df['SW1'] / df['BNR']
            df['CRSWIR'] = df['SW1'] / (df['NIR'] + ((df['SW2'] - df['NIR']) / (2185.7 - 864)) * (1610.4 - 864))
            df['NGRDI'] = (df['GRN'] - df['RED']) / (df['GRN'] + df['RED'])
            df['SRSWIR'] = df['SW1'] / df['SW2']
            ### drop band columns
            df = df.drop(columns=['BLU', 'GRN', 'RED', 'BNR', 'NIR', 'RE1', 'RE2', 'RE3', 'SW1', 'SW2'])
            ### drop the dataframe rows which contain any NaN or NA values
            df.dropna(inplace=True)
            # X = X[['DSWI', 'NDWI', 'TCW', 'CLRE', 'NDREI2', 'NDREI1', 'SWIRI', 'CRSWIR', 'DOY']]
            # ### replace -inf and inf
            # df = df.replace([-np.inf], -1)
            # df = df.replace([np.inf], 1)
            ### to be on the safe side, we drop the rows containing any -inf or inf values as well
            df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
            # ### clip these values to the range -1 to 1
            # df = df.clip(-1, 1)
            # ### clip these values to the range -1 to 1
            # ### make sure to clip only the index columns, not the DOY column
            # ### (also not the QAI column, if it is present)
            # if use_qai == 1:
            #     ### clip all but the last two columns
            #     df.iloc[:, :-2] = df.iloc[:, :-2].clip(-1, 1)
            # else:
            #     ### clip all but the last column
            #     df.iloc[:, :-1] = df.iloc[:, :-1].clip(-1, 1)
            # ### also remove rows with values outside of the range -1 to 1
            # ### make sure to check only the index columns, not the DOY column
            # ### and not the QAI column, if it is present
            # ### remove the row if any index column has an unrealistic value (smaller than -1 or greater than 1)
            # df = df[(df['DSWI'] >= -1) & (df['DSWI'] <= 5)]
            # df = df[(df['NDWI'] >= -1) & (df['NDWI'] <= 1)]
            # # X = X[(X['TCW'] >= -1) & (X['TCW'] <= 1)]
            # df = df[(df['CLRE'] >= -1) & (df['CLRE'] <= 10)]
            # df = df[(df['NDREI2'] >= -1) & (df['NDREI2'] <= 1)]
            # df = df[(df['NDREI1'] >= -1) & (df['NDREI1'] <= 1)]
            # df = df[(df['SWIRI'] >= 0) & (df['SWIRI'] <= 1)] # exception: SWIRI must be positive
            # df = df[(df['CRSWIR'] >= -1) & (df['CRSWIR'] <= 1)]

        return df.mean(skipna=True).values, df.std(skipna=True).values

    ### get mean of each sample for each band separately
    ### see note above: in the end, we take the mean of the band means, 
    ### which is not the same as the mean of all observations
    ### but much faster to process
    results = Parallel(n_jobs=num_workers)(delayed(process_file)(file, use_indices, use_qai, indices_bands) for file in train_files)

    for mean, std in results:
        means.append(mean)
        stds.append(std)

    ### get length of means to check if it worked
    print("Number of samples: ", len(means))

    ### get the mean of the band means
    overall_mean = np.nanmean(means, axis=0)
    overall_std = np.nanmean(stds, axis=0)

    return overall_mean, overall_std


def store_train_val_test(root_dir, num_workers=1, fixed_norm=0, use_indices=0, info="", seed=123, use_qai=0, indices_bands=[]):
    ### set seed
    random.seed(seed)
    np.random.seed(seed)

    ### read metadata
    meta = pd.read_csv(root_dir + '/meta/metadata.csv', sep=";")

    VAL_SPLIT = .2

    if not os.path.exists(os.path.join(root_dir + '/split_data')):
        os.mkdir(os.path.join(root_dir + '/split_data'))
    
    ### some preprocessing steps to avoid mistakes
    meta.dataset = meta.dataset.astype(str)
    meta.loc[meta['dataset'] == 'sax2', 'dataset'] = 'sax'

    ### select only data from specific datasets
    meta = meta[meta['dataset'].isin(
        ['barkbeetle', 'may', 'johmay', 'may_val_shapes', 'may_bark_beetle_val_shapes', 'sax', 'sax2', 
         'rlp', 'fnews', 'bb', 'thu', 'lux', 'schwarz', 'aus', 'lux_healthy', 'fnews_sax', 
         'fnews_bw1', 'fnews_nisa', 'may_val_pts', 'npbf', 'fnews_bw2']
    )]
    # 'schiefer', excluded (cannot be sure about correctness of healthy status, could also be soil or a forest path)
    # may_bark_beetle_val_shapes probably not nessecary because dataset name is may_val_shapes
    # but we want to be on the safe side
    # fnews_bw2 has been deliberately dropped from datasets because it conincides spatially with npbf validation data

    meta.loc[:, 'mort_1'] = meta['mort_1'] + meta['mort_2'] + meta['mort_3'] + meta['mort_4'] \
                            + meta['mort_5'] + meta['mort_6'] + meta['mort_7'] + meta['mort_8'] \
                            + meta['mort_9']
    meta.drop_duplicates(subset='plotID', keep="first", inplace=True)

    ### we need an indication if the pixel is covered merely by deciduous or coniferous trees
    meta['dec_con'] = \
        ['con' if (meta['frac_coniferous'].iloc[x] > .5) else 'dec' for x in range(0, len(meta))]
    

    ### define testdat as all meta rows with dataset == "may"
    ### be aware that there is also a column called "dateset" which is a typo
    ### and contains the values "barkbeetle" for the barkbeetle dataset
    testdat = meta.loc[(meta['dataset'] == 'may') | (meta['dataset'] == 'may_val_shapes') | (meta['dataset'] == 'may_val_pts') | (meta['dataset'] == 'npbf')] #  | (meta['dataset'] == 'fnews_bw2')
    testdat['max_mort'] = ['mort_1' if (testdat['mort_1'].iloc[x] > 0) else 'mort_0' for x in range(0, len(testdat))]

    ### remove those samples from testdat that have dataset == fnews_bw2 and healthy != 1.0
    testdat = testdat.loc[~((testdat['dataset'] == 'fnews_bw2') & (testdat['healthy'] != 1.0))]

    ### remove those rows from meta
    meta = meta.loc[~((meta['dataset'] == 'may') | (meta['dataset'] == 'may_val_shapes') | (meta['dataset'] == 'may_val_pts') | (meta['dataset'] == 'npbf'))] #  | (meta['dataset'] == 'fnews_bw2')

    ### remove the "nrw" dataset from meta to avoid criticism about spatial autocorrelation in paper
    meta = meta.loc[~(meta['dataset'] == 'nrw')]

    ### for testdat, we include mort_cleared and mort_soil to mortality
    ### to avoid false false positives in evaluation
    meta['mort_soil'] = meta['mort_soil'].fillna(0)
    ### drop soil pixels
    meta = meta.loc[~((meta['mort_soil'].fillna(0) + meta['mort_regrowth'].fillna(0)) > 0)]
    ### has been assigned using mort_1-9 already
    meta['mort_1'] = meta['mort_dec'] + meta['mort_con'] + meta['mort_cleared'] # + testdat['mort_soil']
    # clip to 0-1 range
    meta['mort_1'] = meta['mort_1'].clip(0,1)
    # re-compute mort_0 and max_mort
    meta['mort_0'] = 1 - meta['mort_1']
    # re-compute max_mort (we could even set another threshold for testing!?)
    meta['max_mort'] = ['mort_1' if (meta['mort_1'].iloc[x] > 0) else 'mort_0' for x in range(0, len(meta))]

    ### lux_healthy dataset contains NaN values in "healthy" and mort_0 column
    ### we need to set them to 1 because otherwise they would be removed
    ### but only for the subset meta[meta['dataset'] == 'lux_healthy']
    ### note that I don't know why mort_0 would be NaN, because it has correctly been assigned in preparation script
    ### anyway, we re-assign it here
    meta.loc[(meta['dataset'] == 'lux_healthy') & (meta['healthy'].isnull()), 'healthy'] = 1.0
    meta.loc[(meta['dataset'] == 'lux_healthy') & (meta['mort_0'].isnull()), 'mort_0'] = 1.0


    ### remove rows of meta which contain meta['dataset'] == "fnews" AND meta["healthy"] != 1
    ### these are the fnews samples of which we cannot tell for sure that they are completely undisturbed
    ### because they are at the edge of the healthy polygons
    # fnews = meta.loc[(meta['dataset'].isin(['fnews']))]
    # meta = meta.loc[~(meta['dataset'].isin(['fnews']))]
    meta = meta.loc[~((meta['dataset'].str.contains('fnews')) & (meta['healthy'] != 1))]
    ### assign 1.0 to all mort_0 values of mort["dataset"] == "fnews"
    meta.loc[(meta['dataset'].str.contains('fnews')), 'mort_0'] = 1.0

    ### remove all rows of meta with mort_0 != 0
    meta = meta.loc[meta['mort_0'] == 1.0]

    ### remove all rows of meta with dec_con == 'dec'
    meta = meta.loc[meta['dec_con'] == 'con'] 
    meta = meta.loc[(meta['frac_coniferous'] > 0.5)]
    ### for the time we comment out this line to include all samples
    ### because we need the model to be able to cope with mixed forest as well
    ### and thus, only training on pure coniferous pixels could lead to unreasonable behaviour

    ### note that later it might be necessary to over- or undersample some group
    ### so that dec and con are balanced in the training set

    ### split the data into train and val
    ### we opt for random split between all data (except test data)
    ### because LUX and FNEWS data usually has less years than the Schiller AOI's
    ### and we want avoid biases!
    traindat, valdat = train_test_split(meta, test_size=VAL_SPLIT, random_state=seed)
    # ### instead of this random split, use all datasets "aus", "sax", "thu", "rlp", "bb" as valdat
    # ### and everything else as traindat
    # valdat = meta.loc[meta['dataset'].isin(['aus', 'sax', 'thu', 'rlp', 'bb'])]
    # traindat = meta.loc[~meta['dataset'].isin(['aus', 'sax', 'thu', 'rlp', 'bb'])]

    ### print number of samples in each set
    print("\nTraining Set Size: ", len(traindat))
    print("\nValidation Set Size: ", len(valdat))
    print("\nTest Set Size: ", len(testdat))

    ### save the split data incl args.info as for the mean and sd parameters
    traindat.to_csv(root_dir + '/split_data/train_' + str(info) + '.csv', index=False)
    valdat.to_csv(root_dir + '/split_data/val_' + str(info) + '.csv', index=False)
    testdat.to_csv(root_dir + '/split_data/test_' + str(info) + '.csv', index=False)

    if fixed_norm == 1:
        print("Computing fixed normalization parameters...")
        ### compute fixed normalization parameters
        train_files = [root_dir + "/" + f + ".csv" for f in traindat['plotID']]
        overall_mean, overall_std = calculate_normalization_params(train_files, num_workers, use_indices, use_qai, indices_bands)
        ### convert to DataFrame
        mean_df = pd.DataFrame(overall_mean, columns=['mean'])
        std_df = pd.DataFrame(overall_std, columns=['std'])
        if len(indices_bands) > 0:
            mean_df.to_csv(root_dir + '/split_data/train_overall_mean_indices_bands_' + str(info) + '.csv', index=False)
            std_df.to_csv(root_dir + '/split_data/train_overall_std_indices_bands_' + str(info) + '.csv', index=False)
        elif use_indices == 1:
            ### save the normalization parameters
            # np.save(root_dir + '/split_data/train_overall_mean_indices_' + str(info) + '.npy', overall_mean)
            # np.save(root_dir + '/split_data/train_overall_std_indices_' + str(info) + '.npy', overall_std)
            mean_df.to_csv(root_dir + '/split_data/train_overall_mean_indices_' + str(info) + '.csv', index=False)
            std_df.to_csv(root_dir + '/split_data/train_overall_std_indices_' + str(info) + '.csv', index=False)
        else:
            ### save the normalization parameters
            # np.save(root_dir + '/split_data/train_overall_mean_' + str(info) + '.npy', overall_mean)
            # np.save(root_dir + '/split_data/train_overall_std_' + str(info) + '.npy', overall_std)
            mean_df.to_csv(root_dir + '/split_data/train_overall_mean_' + str(info) + '.csv', index=False)
            std_df.to_csv(root_dir + '/split_data/train_overall_std_' + str(info) + '.csv', index=False)

    ### return the csv files' names
    return traindat["plotID"], valdat["plotID"], testdat["plotID"]