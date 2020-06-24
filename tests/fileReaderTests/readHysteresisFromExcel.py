#!python3

from magLabUtilities.datafileutilities.columnData import importFromXlsx

if __name__ == '__main__':
    fp = './tests/fileReaderTests/M(H)_Curve.xlsx'
    fileSignalBundle = importFromXlsx.columnHeadersFromFile(fp, '21k', 3, 'C,D,E', excelTColumn='B', dataColumnNames=['Magnetization', 'Magnetic Field', 'Susceptibility'])

    print('done')
