#!/usr/bin/env python3

import logging

try:
    import coloredlogs
    coloredlogs.install(level='DEBUG')
except ImportError:
    pass
logger = logging.getLogger(__name__)

import os
import argparse

logger.debug("Import pandas and numpy")
import pandas as pd
import numpy as np

mapping = {
    'SPOL': {
        'men': 'MEN',
        'women': 'WOMEN',
    },
    'REZIDENT': {
        'resident': 'RES',
        'mobile': 'MOB',
    },
    'KLS_P16_OPIS_ANGL_1R': {
        'Education': 'FOE01',
        'Arts and humanities': 'FOE02',
        'Social sciences, journalism and information': 'FOE03',
        'Business, administration and law': 'FOE04',
        'Natural sciences, mathematics and statistics': 'FOE05',
        'Information and Communication Technologies (ICTs)': 'FOE06',
        'Engineering, manufacturing and construction': 'FOE07',
        'Agriculture, forestry, fisheries and veterinary': 'FOE08',
        'Health and welfare': 'FOE09',
        'Services': 'FOE10',
    },
    'NACIN_STUDIJA': {
        'PART-TIME': 'PARTTIME',
        'FULL-TIME': 'FULLTIME',
    },
    'STAROST_OBMOCJA': {
        'under 20 years old': 'BELOW20',
        'between 20 and 21 years old': '20TO21',
        'between 22 and 24 years old': '22TO24',
        'between 25 and 29 years old': '25TO29',
        'over 29 years old': 'OVER29',
    },
}

flag_suffix = {
    'SPOL': 'GEN',
    'REZIDENT': 'MOB',
    'KLS_P16_OPIS_ANGL_1R': 'FOE',
    'NACIN_STUDIJA': 'PARTFULL',
    'STAROST_OBMOCJA': 'AGE',
}

map_levels = {
    'ISCED6': 6,
    'ISCED7': "7 - master",
    'ISCED7LONG': "7 - long degree",
    'ISCED8': 8,
}

prefix = {
    'STUD': {
        'ISCED6': 'STUD{sep}',
        'ISCED7': 'STUD{sep}',
        'ISCED7LONG': 'STUD{sep}',
        'ISCED8': 'RES{sep}STUD',
    },
    'GRAD': {
        'ISCED6': 'GRAD{sep}',
        'ISCED7': 'GRAD{sep}',
        'ISCED7LONG': 'GRAD{sep}',
        'ISCED8': 'RES{sep}GRAD',
    },
}


eVS_kwargs = {
    'dtype': {
        'STUDIJSKO_LETO': pd.CategoricalDtype([2017, 2018, 2019, 2020, 2021, 2022, 2023], ordered=True),
        'SIFRA_ZAVODA': 'Int16',
        'UNIVERZA': 'string',
        'NAZIV_ZAVODA_ANGL_NVL': 'string',
        'SPOL': pd.CategoricalDtype(mapping['SPOL'].keys(), ordered=True),
        'REZIDENT': pd.CategoricalDtype(mapping['REZIDENT'].keys(), ordered=True),
        'MOBILNOST_STUDIJA': pd.CategoricalDtype(['resident', 'mobile']),
        'ISCED_VREDNOST': pd.CategoricalDtype(map_levels.values(), ordered=True),
        'KLS_P16_OPIS_ANGL_1R': pd.CategoricalDtype(mapping['KLS_P16_OPIS_ANGL_1R'].keys(), ordered=True),
        'NACIN_STUDIJA': pd.CategoricalDtype(mapping['NACIN_STUDIJA'].keys(), ordered=True),
        'STAROST_OBMOCJA': pd.CategoricalDtype(mapping['STAROST_OBMOCJA'].keys(), ordered=True),
        'ST': 'Int64',
    }
}

orgreg_kwargs = {
    'dtype': {
        'ETER_ID': 'string',
        'Name_Orgreg': 'string',
        'NID': 'int16',
        'ROR_ID': 'string',
        'WHED_ID': 'string',
        'DEQAR_ID': 'string',
        'Erasmus_code': 'string',
    },
    'index_col': 'NID',
}

anonymisation_threshold = 5

def f_year_level(df, year, level):
    """
    select rows for relevant academic year and ISCED level
    """
    return df[(df.STUDIJSKO_LETO == year) & (df.ISCED_VREDNOST == map_levels.get(level))]

def anonymise_row(row, flags):
    """
    anonymise values in a row if below threshold
    """
    if (row[row.index != flags] <= anonymisation_threshold).any():
        anon = row.map(lambda n: 'm')
        anon[flags] = 'c'
        return anon
    else:
        return row

def do_breakdown(df, year, category, level, char):
    """
    do breakdown by specific characteristic
    """
    prefix_data = prefix[category][level].format(sep='.')
    flag_column = prefix[category][level].format(sep='.FLAG') + level + flag_suffix[char]
    breakdown = df.groupby(['SIFRA_ZAVODA', char], observed=False)['ST'] \
        .sum() \
        .unstack() \
        .rename(columns={ k: f'{prefix_data}{level}{v}' for k, v in mapping.get(char).items() })
    breakdown[flag_column] = ''
    return breakdown.transform(anonymise_row, axis='columns', flags=flag_column)

def do_year(df, year, category):
    """
    combine breakdowns by levels and otehr characteristics
    """
    frames = []
    for level in map_levels.keys():
        selection = df.pipe(f_year_level, year, level)
        for char in mapping.keys():
            if char in df.columns:
                frames.append(selection.pipe(do_breakdown, year, category, level, char))
        total_data = prefix[category][level].format(sep='.') + level + 'TOTAL'
        total_flag = prefix[category][level].format(sep='.FLAG') + level + 'TOTAL'
        totals = pd.DataFrame({
            total_data: selection.groupby(['SIFRA_ZAVODA'], observed=False)['ST'].sum(),
            total_flag: ''
        })
        frames.append(totals.transform(anonymise_row, axis='columns', flags=total_flag))
    return pd.concat(frames, axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("SRC", help="source file")
    parser.add_argument("YEAR", help="academic year(s)", type=int, nargs='+')
    parser.add_argument("-d", "--destination", help="output directory for generated files", default='.')
    parser.add_argument("-o", "--orgreg", help="files with OrgReg reference data", default='orgreg.ods')
    parser.add_argument("-c", "--category", help="students or graduates", type=str.upper, default='STUD', choices=['STUD','GRAD'])
    args = parser.parse_args()

    if not (os.path.isdir(args.destination) and os.access(args.destination, os.W_OK)):
        raise Exception(f"{args.destination} is not a valid and writable path")

    logger.info(f"Loading eVS data from {args.SRC}")
    eVS = pd.read_excel(args.SRC, **eVS_kwargs)

    logger.info(f"Loading OrgReg reference data from {args.orgreg}")
    orgreg = pd.read_excel(args.orgreg, **orgreg_kwargs)

    for year in args.YEAR:
        logger.info(f"Exporting tables for {year}")
        this = eVS.pipe(do_year, year, args.category).fillna(0).join(orgreg).sort_values('ETER_ID') #.map(lambda x: int() if pd.isna(x) else x)
        this['BAS.ETERIDYEAR'] = this['ETER_ID'] + f'.{year}'
        this.set_index('BAS.ETERIDYEAR').to_excel(os.path.join(args.destination, f'ETER_{os.path.splitext(os.path.basename(args.SRC))[0]}_{year}.ods'), sheet_name='additional-data')

