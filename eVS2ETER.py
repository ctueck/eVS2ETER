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

add_blank = {
    'SPOL': [
		'GENUNCL',
        'GEN_FLAG',
		'NAT',
		'FOR',
		'CITUNCL',
        'CITIZ_FLAG',
    ],
    'REZIDENT': [
		'MOBUNCL',
        'MOB_FLAG',
		'FOE00',
    ],
    'KLS_P16_OPIS_ANGL_1R': [
		'FOEUNCL',
        'FOE_FLAG',
    ],
    'NACIN_STUDIJA': [
		'PARTFULLUNCL',
        'PARTFULL_FLAG',
    ],
    'STAROST_OBMOCJA': [
		'AGEUNCL',
        'AGE_FLAG',
    ],
    'TOTAL': [
        'TOTAL_FLAG',
        'NOTES',
    ],
}

map_levels = {
    'ISCED6': 6,
    'ISCED7': "7 - master",
    'ISCED7LON': "7 - long degree",
    'ISCED8': 8,
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

def f_year_level(df, year, level):
    """
    select rows for relevant academic year and ISCED level
    """
    return df[(df.STUDIJSKO_LETO == year) & (df.ISCED_VREDNOST == map_levels.get(level))]

def do_breakdown(df, char):
    """
    do breakdown by specific characteristic
    """
    return df.groupby(['SIFRA_ZAVODA', char], observed=False)['ST'].sum().unstack().rename(columns=mapping.get(char))

def do_year(df, year):
    """
    combine breakdowns by levels and otehr characteristics
    """
    frames = []
    for level in map_levels.keys():
        selection = df.pipe(f_year_level, year, level)
        for char in mapping.keys():
            if char in df.columns:
                this_df = selection.pipe(do_breakdown, char).rename(columns=lambda n: f'STUD.{level}{n}')
                for blank in add_blank[char]:
                    this_df[f'STUD.{level}{blank}'] = pd.NA
                frames.append(this_df)
        total = pd.DataFrame({
            f'STUD.{level}TOTAL': selection.groupby(['SIFRA_ZAVODA'], observed=False)['ST'].sum(),
        })
        for blank in add_blank['TOTAL']:
            total[f'STUD.{level}{blank}'] = pd.NA
        frames.append(total)
    return pd.concat(frames, axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("SRC", help="source file")
    parser.add_argument("YEAR", help="academic year(s)", type=int, nargs='+')
    parser.add_argument("-d", "--destination", help="output directory for generated files", default='.')
    parser.add_argument("-o", "--orgreg", help="files with OrgReg reference data", default='orgreg.ods')
    args = parser.parse_args()

    if not (os.path.isdir(args.destination) and os.access(args.destination, os.W_OK)):
        raise Exception(f"{args.destination} is not a valid and writable path")

    logger.info(f"Loading eVS data from {args.SRC}")
    eVS = pd.read_excel(args.SRC, **eVS_kwargs)

    logger.info(f"Loading OrgReg reference data from {args.orgreg}")
    orgreg = pd.read_excel('orgreg.ods', **orgreg_kwargs)

    for year in args.YEAR:
        logger.info(f"Exporting tables for {year}")
        eVS.pipe(do_year, year).join(orgreg).sort_values('ETER_ID').to_excel(os.path.join(args.destination, f'ETER_{os.path.splitext(os.path.basename(args.SRC))[0]}_{year}.ods'))

# TO DO for future ref:
#
# - fill cells with 0
#

