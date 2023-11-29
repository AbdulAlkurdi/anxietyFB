import process_redcap
import json
import sys
from pathlib import Path
# pylint: disable=pointless-string-statement

def create_radwear_metadata(
    radwear_path='/mnt/c/Users/alkurdi/Desktop/Vansh/data/RADWear/', force_update=False
):
    """
    Creates metadata for RADWear participants.

    Args:
        radwear_path (str): Path to the RADWear data directory. Default is '/mnt/c/Users/alkurdi/Desktop/Vansh/data/RADWear/'.
        force_update (bool): Flag to force update the metadata even if it already exists. Default is False.
    """
    redcap_path = radwear_path + 'REDCap responses/'

    my_file = Path(radwear_path + 'all_p_metadata.json')
    if my_file.is_file() and not force_update:
        print('metadata file exists')
        with open(radwear_path + 'all_p_metadata.json', 'rb') as f:
            all_p_metadata = json.load(f)
            print('json loaded')
    else:
        if force_update:
            print('metadata data exists but update forced')
        else:
            print('metadata data is not ready. will be created')

        participants = [4, 7, 9, 12, 14, 16, 17, 18, 21]  # dropped 5, 20

        status = {
            4: 'calibration only',
            5: 'disregard, missing many E4 + noisy',
            7: 'complete',
            9: 'complete',
            12: 'complete',
            14: 'complete',
            16: 'ongoing',
            17: 'complete',
            18: 'calibration only',
            20: 'disregard, calibration only and missing e4',
            21: 'ongoing',
        }

        '''
        keys for data:
        low anxiety: LA
        high anxiety: HA
        e4sn: e4 serial number
        hxsn: hexoskin serial number
        '''

        # participant days: participant_id: [number of days for low anxiety, number of days for high anxiety]
        partcipant_days = {
            4: [0, 0],
            5: [0, 8],
            7: [11, 10],
            9: [10, 10],
            12: [8, 10],
            14: [9, 10],
            16: [9, 0],
            17: [10, 10],
            18: [0, 0],
            20: [0, 0],
            21: [9, 0],
        }

        total = 0
        for each in list(partcipant_days.values()):
            total += each[0]
            total += each[1]
        print(
            'number of days total of redcap data available for RADWear in the wild: ',
            total,
        )

        # p4
        p4_E4_calib = '220714-232953'
        p4_hx_calib = '248835'

        p4_E4_LA = []
        p4_hx_LA = []
        p4_E4_HA = []
        p4_hx_HA = []

        p4_redcap_avail = [[], []]
        p4 = {
            'status': status[4],
            'e4sn': 'A0381C',
            'hxsn': '4772',
            'complete days': partcipant_days[4],
            'RedCap available': p4_redcap_avail,
            'calibration': [p4_E4_calib, p4_hx_calib],
            'LA': [p4_E4_LA, p4_hx_LA],
            'HA': [p4_E4_HA, p4_hx_HA],
        }

        # p5

        p5_E4_calib = '220704-164707'
        p5_hx_calib = '250343'

        p5_E4_LA = []
        p5_hx_LA = []
        p5_E4_HA = [
            '220927-115055',
            '220928-115104',
            '220929-114633',
            '220930-114429',
            '221001-120807',
            '221003-114642',
            '221004-120334',
            '221005-110525' + '221005-145538',
        ]
        p5_hx_HA = [
            '253647',
            '253648',
            '253649',
            '253650',
            '253651',
            '253652',
            '253753',
            '254429',
        ]
        p5_redcap_avail = [[], [1, 1, 1, 1, 1, 1, 1, 0]]

        # p5_redcap = [all_participants_redcap_dict[5]] # not done yet

        p5 = {
            'status': status[5],
            'e4sn': 'A037F7',
            'hxsn': '8783',
            'complete days': partcipant_days[5],
            'RedCap available': p5_redcap_avail,
            'calibration': [p5_E4_calib, p5_hx_calib],
            'LA': [p5_E4_LA, p5_hx_LA],
            'HA': [p5_E4_HA, p5_hx_HA],
        }

        # p7

        p7_E4_calib = '220712-181114'
        p7_hx_calib = '248582'

        p7_E4_LA = [
            '220829-121320',
            '220830-114442',
            '220831-120141',
            '220901-113936',
            '220902-114700',
            '220906-113947',
            '220907-120003',
            '220908-120451',
            '220909-120309',
            '220912-151726',
            '220913-134458',
        ]
        p7_hx_LA = [
            0,
            '251292',
            '251331',
            '251374',
            '251399',
            '251559',
            '251577',
            '251622',
            '251779',
            '251928',
            '251984',
        ]

        p7_E4_HA = [
            '230410-115059',
            '230411-122423',
            '230412-121912',
            '230413-123640',
            '230414-122016',
            0,
            '230418-123119',
            '230419-121005',
            '230420-144405',
            '230421-163430',
        ]
        p7_hx_HA = [
            '264721',
            '264747',
            '264775',
            '264809',
            '264836',
            '265005',
            '265006',
            '265069',
            '265174',
            '265224',
        ]

        p7_redcap_avail = [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]  # ?, ?]
        # p7_redcap = [all_participants_redcap_dict[7]] # not done yet

        p7 = {
            'status': status[7],
            'e4sn': 'A04C05',
            'hxsn': '8550',
            'complete days': partcipant_days[7],
            'RedCap available': p7_redcap_avail,
            'calibration': [p7_E4_calib, p7_hx_calib],
            'LA': [p7_E4_LA, p7_hx_LA],
            'HA': [p7_E4_HA, p7_hx_HA],
        }

        # p9

        p9_E4_calib = '220704-164707'
        p9_hx_calib = '248258'

        p9_E4_LA = [
            '220801-180621',
            '220802-163608',
            '220803-145947',
            '220804-151805',
            '220805-134635',
            '220808-145300',
            '220809-142505',
            '220810-152951',
            '220811-161838',
            '220812-154403',
        ]
        p9_hx_LA = [
            '249726',
            '249805',
            '249918',
            '249974',
            '250026',
            '250069',
            '250118',
            '250325',
            '250410',
            '250464',
        ]
        p9_E4_HA = [
            '220815-122425',
            '220816-114335',
            '220817-113455',
            '220818-120324',
            '220819-114116',
            '220821-214544',
            '220822-223419',
            '220823-222539',
            '220824-230220',
            '220825-232501' + '220826-071128',
        ]
        p9_hx_HA = [
            '250527',
            '250598',
            '250774',
            '250850',
            '250875',
            '250947',
            '250977',
            '251011',
            '251094',
            '251136',
        ]

        p9_redcap_avail = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]  # not done yet
        # p9_redcap = [all_participants_redcap_dict[9]]

        p9 = {
            'status': status[9],
            'e4sn': 'A04BA8',
            'hxsn': '8872',
            'complete days': partcipant_days[9],
            'RedCap available': p9_redcap_avail,
            'calibration': [p9_E4_calib, p9_hx_calib],
            'LA': [p9_E4_LA, p9_hx_LA],
            'HA': [p9_E4_HA, p9_hx_HA],
        }

        # p12

        p12_E4_calib = '220701-002248'
        p12_hx_calib = '248092'

        p12_E4_LA = [
            '220718-132014',
            '220719-130314',
            '220720-120906',
            '220721-122543',
            '220725-121303',
            '220726-123528',
            '220727-121705',
            '220801-120508',
        ]
        p12_hx_LA = [
            '248962',
            '249050',
            '249152',
            '249272',
            '249373',
            '249417',
            '249468',
            '251204',
        ]
        p12_E4_HA = [
            '220911-220003',
            '220912-222451',
            '220913-222507',
            '220915-221059',
            '220916-060310',
            '220919-112244',
            '220920-112551',
            '220921-122019',
            '220922-113528',
            '220923-113640',
        ]
        p12_hx_HA = [
            '251915',
            '251983',
            '252042',
            '252251',
            '252252',
            '252315',
            '252395',
            '252522',
            '252609',
            '252733',
        ]

        p12_redcap_avail = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]  # redcap data is shifted and dates aren't all correct
        # p12_redcap = [all_participants_redcap_dict[12]]

        p12 = {
            'status': status[12],
            'e4sn': 'A038E2',
            'hxsn': '7234',
            'complete days': partcipant_days[12],
            'RedCap available': p12_redcap_avail,
            'calibration': [p12_E4_calib, p12_hx_calib],
            'LA': [p12_E4_LA, p12_hx_LA],
            'HA': [p12_E4_HA, p12_hx_HA],
        }

        # p14

        p14_E4_calib = '220620-224412'
        p14_hx_calib = '247555'

        p14_E4_LA = [
            '220718-115638',
            '220719-160849',
            '220720-163536',
            '220721-125931',
            '220722-142715',
            '220726-133230',
            0,
            '220728-121347',
            '220729-112349',
        ]
        p14_hx_LA = [
            '248966',
            '249044',
            '249151',
            '249201',
            '249278',
            '249406',
            '249469',
            '249524',
            '249561',
        ]
        p14_E4_HA = [
            '230501-113540',
            '230502-142834',
            '230503-115351',
            '230504-115657',
            '230505-115115',
            '230508-113742',
            '230509-133128',
            '230510-122315',
            '230511-125537',
            '230512-114245',
        ]
        p14_hx_HA = [
            0,
            '265813',
            '265915',
            '266004',
            '266106',
            '266187',
            '266302',
            '266365',
            '266415',
            '266815',
        ]

        p14_redcap_avail = [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]]
        # p14_redcap = [all_participants_redcap_dict[14]]

        p14 = {
            'status': status[14],
            'e4sn': 'A036EE',
            'hxsn': '44675',
            'complete days': partcipant_days[14],
            'RedCap available': p14_redcap_avail,
            'calibration': [p14_E4_calib, p14_hx_calib],
            'LA': [p14_E4_LA, p14_hx_LA],
            'HA': [p14_E4_HA, p14_hx_HA],
        }

        # p16

        p16_E4_calib = '230727-000346'
        p16_hx_calib = '270761'

        p16_E4_LA = [
            '230828-130409',
            '230829-131230',
            '230830-130830',
            '230831-124044',
            '230901-124432',
            '230905-132521',
            '230906-131815',
            '230907-125434',
            '230908-131813',
        ]
        p16_hx_LA = [
            '272490',
            '272532',
            '272876',
            '272877',
            '272878',
            '272931',
            '273035',
            '273155',
            '273236',
        ]
        p16_E4_HA = []
        p16_hx_HA = []

        p16_redcap_avail = [[1, 1, 1, 1, 1, 1, 1, 1, 1], []]
        # p16_redcap = [all_participants_redcap_dict[16]]

        p16 = {
            'status': status[16],
            'e4sn': 'A0343F',
            'hxsn': '8783',
            'complete days': partcipant_days[16],
            'RedCap available': p16_redcap_avail,
            'calibration': [p16_E4_calib, p16_hx_calib],
            'LA': [p16_E4_LA, p16_hx_LA],
            'HA': [p16_E4_HA, p16_hx_HA],
        }

        # p17

        p17_E4_calib = '230727-000250'
        p17_hx_calib = '270766'

        p17_E4_LA = [
            '230807-125842',
            '230808-132814',
            '230809-134843',
            '230810-131337',
            '230811-104045',
            '230814-102700',
            '230815-113210',
            '230816-110843',
            0,
            '230818-112608',
        ]
        p17_hx_LA = [
            '271384',
            '271418',
            '271455',
            '271493',
            '271514',
            '271596',
            '271644',
            '271739',
            '271784',
            '271833',
        ]
        p17_E4_HA = [
            '230821-104414',
            '230822-133555',
            '230823-105017',
            '230824-102510',
            '230825-113427',
            '230828-114102',
            '230829-220819',
            '230830-101242',
            '230831-101208',
            '230901-113018',
        ]
        p17_hx_HA = [
            '272004',
            '272113',
            '272204',
            '272304',
            '272422',
            '272489',
            '272536',
            '272647',
            '272730',
            '272804',
        ]

        p17_redcap_avail = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        # p17_redcap = [all_participants_redcap_dict[17]]

        p17 = {
            'status': status[17],
            'e4sn': 'A0340C',
            'hxsn': '8536',
            'complete days': partcipant_days[17],
            'RedCap available': p17_redcap_avail,
            'calibration': [p17_E4_calib, p17_hx_calib],
            'LA': [p17_E4_LA, p17_hx_LA],
            'HA': [p17_E4_HA, p17_hx_HA],
        }

        # p18

        p18_E4_calib = '230727-000306'
        p18_hx_calib = '270759'

        p18_E4_LA = []
        p18_hx_LA = []
        p18_E4_HA = []
        p18_hx_HA = []

        # p18_redcap_avail = [all_participants_redcap_dict[18]]
        p18_redcap_avail = [[], []]

        p18 = {
            'status': status[18],
            'e4sn': 'A038E2',
            'hxsn': '',
            'complete days': partcipant_days[18],
            'RedCap available': p18_redcap_avail,
            'calibration': [p18_E4_calib, p18_hx_calib],
            'LA': [p18_E4_LA, p18_hx_LA],
            'HA': [p18_E4_HA, p18_hx_HA],
        }
        # p20

        p20_E4_calib = ''
        p20_hx_calib = '270760'

        p20_E4_LA = []
        p20_hx_LA = []
        p20_E4_HA = []
        p20_hx_HA = []

        # p20_redcap_avail = [all_participants_redcap_dict[20]]
        p20_redcap_avail = []

        p20 = {
            'status': status[20],
            'e4sn': '',
            'hxsn': '',
            'complete days': partcipant_days[20],
            'RedCap available': p20_redcap_avail,
            'calibration': [p20_E4_calib, p20_hx_calib],
            'LA': [p20_E4_LA, p20_hx_LA],
            'HA': [p20_E4_HA, p20_hx_HA],
        }

        # p21

        p21_E4_calib = '230727-000318'
        p21_hx_calib = '270764'

        p21_E4_LA = [
            '230821-124927',
            '230822-160632',
            '230823-120957',
            '230825-124152',
            '230828-114504',
            '230829-114425',
            '230830-114324',
            '230831-114743',
            '230901-120509',
            '230907-170014',
        ]
        p21_hx_LA = [
            '272105',
            '272124',
            '272189',
            '272424',
            '272482',
            '272535',
            '272628',
            '272765',
            '272806',
            '273162',
        ]
        p21_E4_HA = []
        p21_hx_HA = []

        p21_redcap_avail = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], []]
        # p21_redcap = [all_participants_redcap_dict[21]]

        p21 = {
            'status': status[21],
            'e4sn': 'A04C05',
            'hxsn': '8550',
            'complete days': partcipant_days[21],
            'RedCap available': p21_redcap_avail,
            'calibration': [p21_E4_calib, p21_hx_calib],
            'LA': [p21_E4_LA, p21_hx_LA],
            'HA': [p21_E4_HA, p21_hx_HA],
        }
        fs = {
            'ECG': 256,
            'BVP': 64,
            'BR': 1,
            'TEMP': 4,
            'EDA': 4,
            'ACC_hx': 64,
            'ACC_e4': 32,
        }
        all_p_metadata = {
            'list of participant IDs': participants,
            4: p4,
            5: p5,
            7: p7,
            9: p9,
            12: p12,
            14: p14,
            16: p16,
            17: p17,
            18: p18,
            20: p20,
            21: p21,
            'fs': fs,
        }

        with open(radwear_path + 'all_p_metadata.json', 'w') as fp:
            json.dump(all_p_metadata, fp)
        print('metadata file created')


def create_wear_metadata(
    wear_path='/mnt/c/Users/alkurdi/Desktop/Vansh/data/Wear/', force_update=False
):  # need to change wear path probably
    """
    Creates metadata for the WEAR project.

    Args:
        wear_path (str): The path to the WEAR project directory. Default is '/mnt/c/Users/alkurdi/Desktop/Vansh/data/Wear/'.
        force_update (bool): If True, forces the update of metadata even if it already exists. Default is False.
    """
    '''
    NOTES:
    [Box Health - Internal] Rad-Wear/WEAR data analysis is the file that contains the tags 
    for the participants that mark each of the events during their in-lab sessions.
    '''  # statement: ignore

    
    redcap_path = (
        wear_path + 'REDCap responses/'
    )  # probably need to change based on where it actually is

    my_file = Path(wear_path + 'all_p_metadata.json')  # don't know what the file is...
    if my_file.is_file() and not force_update:
        print('metadata file exists')
        with open(wear_path + 'all_p_metadata.json', 'rb') as f:
            all_p_metadata = json.load(f)
            print('json loaded')
    else:
        if force_update:
            print('metadata data exists but update forced')
        else:
            print('metadata data is not ready. will be created')

        participants = [1, 5, 8, 9, 13, 15, 24, 29, 33, 39, 92, 216]

        status = {
            1: "complete",
            5: "missing many E4 and Hx",
            8: "ongoing",
            9: "complete",
            13: "complete",
            15: "complete",
            24: "complete",
            29: "complete",
            33: "ongoing",
            39: "complete",
            92: "complete, missing 1 Hx",
            216: "complete",
        }

        '''
        keys for data:
        e4sn: e4 serial number
        hxsn: hexoskin serial number
        '''

        # participant days: participant_id: number of days total (based on sheet, counted only days with both E4 and hx)

        # !! need to add P8 and P33 once information is updated on sheet
        partcipant_days = {
            1: 10,
            5: 4,
            9: 10,
            13: 10,
            15: 10,
            24: 10,
            29: 10,
            39: 10,
            92: 8,
            216: 10,
        }
        total = 0
        for each in list(partcipant_days.values()):
            total += each
        print(
            'number of days total of redcap data available for Wear in the wild: ',
            total,
        )

        # p1

        # Unsure of calibration files
        p1_e4_calib = ''  # !! need to add
        p1_hx_calib = ''  # !! need to add

        p1_redcap_avail = [[], []]  # !! need to add
        p1_e4 = [
            '230404-132356',
            '230405-131309',
            '230406-141905',
            '230407-132251',
            '230408-184742',
            '230411-132517',
            '230412-142904',
            '230413-140338',
            '230414-134109',
            '230415-183909',
        ]

        p1_hx = [
            '264483',
            '264697',
            '264698',
            '264699',
            '264700',
            '264858',
            '264859',
            '264860',
            '264861',
            '264862',
        ]
        p1 = {
            'status': status[1],
            'e4sn': 'A04BA8',
            'hxsn': '44577',
            'complete days': partcipant_days[1],
            'RedCap available': p1_redcap_avail,
            'calibration': [p1_e4_calib, p1_hx_calib],
            'files': [p1_e4, p1_hx],  # changed name because there isn't LA vs HA
        }

        # p5

        p5_e4_calib = ''  # !! need to add
        p5_hx_calib = ''  # !! need to add

        p5_E4 = ['230426-135632', '230427-152358', '230429-145211', '230502-163446']

        p5_hx = ['265560', '265595', '265682', '265696']

        p5_redcap_avail = []  # !! need to add

        p5 = {
            'status': status[5],
            'e4sn': 'A0343F',
            'hxsn': '7234',
            'complete days': partcipant_days[5],
            'RedCap available': p5_redcap_avail,
            'calibration': [p5_e4_calib, p5_hx_calib],
            'files': [p5_E4, p5_hx],
        }

        # p8 - need to add E4 and hx files

        p8_e4_calib = ''  # !! need to add
        p8_hx_calib = ''  # !! need to add

        p8_e4 = []  # !! need to add
        p8_hx = []  # !! need to add

        p8_redcap_avail = []  # !! need to add

        p8 = {
            'status': status[8],
            'e4sn': 'A038E2',
            'hxsn': '8874',
            'complete days': partcipant_days[8],
            'RedCap available': p8_redcap_avail,
            'calibration': [p8_e4_calib, p8_hx_calib],
            'files': [p8_e4, p8_hx],
        }

        # p9
        p9_E4_calib = ''  # !! need to add
        p9_hx_calib = ''  # !! need to add

        p9_e4 = [
            '230425-143228',
            '230426-165559',
            '230427-143616',
            '230428-174536',
            '230429-142458',
            '230502-142825',
            '230503-173527',
            '230504-170932',
            '230505-180707',
            '230506-183419',
        ]
        p9_hx = [
            '265474',
            '265559',
            '265596',
            '265670',
            '265679',
            '265805',
            '265969',
            '266035',
            '266107',
            '266120',
        ]

        p9_redcap_avail = []  # !! need to add

        p9 = {
            'status': status[9],
            'e4sn': 'A04BA8',
            'hxsn': '8783',
            'complete days': partcipant_days[9],
            'RedCap available': p9_redcap_avail,
            'calibration': [p9_E4_calib, p9_hx_calib],
            'files': [p9_e4, p9_hx],
        }

        # p13

        p13_E4_calib = ''  # !! need to add
        p13_hx_calib = ''  # !! need to add

        p13_E4 = [
            '230404-115842',
            '230405-132007',
            '230406-142342',
            '230407-131855',
            '230408-153018',
            '230411-121507',
            '230412-131412',
            '230413-143014',
            '230414-132840',
            '230415-144249',
        ]

        p13_hx = [
            '264482',
            '264515',
            '264576',
            '264662',
            '264681',
            '264745',
            '264776',
            '264805',
            '264837',
            '264846',
        ]

        p13_redcap_avail = []  # !! need to add

        p13 = {
            'status': status[13],
            'e4sn': 'A0343F',
            'hxsn': '7234',
            'complete days': partcipant_days[13],
            'RedCap available': p13_redcap_avail,
            'calibration': [p13_E4_calib, p13_hx_calib],
            'files': [p13_E4, p13_hx],
        }

        # p15

        p15_E4_calib = ''  # !! need to add
        p15_hx_calib = ''  # !! need to add

        p15_E4 = [
            '230411-205830',
            '230412-114653',
            '230413-124612',
            '230414-131100',
            '230415-143319',
            '230418-142044',
            '230419-120502',
            '230420-115545',
            '230421-130842',
            '230429-155212',
        ]
        p15_hx = [
            '264750',
            '264777',
            '264808',
            '264838',
            '264845',
            '265004',
            '265065',
            '265177',
            '265227',
            '265683',
        ]

        p15_redcap_avail = []  # !! need to add

        p15 = {
            'status': status[15],
            'e4sn': 'A038E2',
            'hxsn': '8774',
            'complete days': partcipant_days[15],
            'RedCap available': p15_redcap_avail,
            'calibration': [p15_E4_calib, p15_hx_calib],
            'files': [p15_E4, p15_hx],
        }

        # p24

        p24_E4_calib = ''  # !! need to add
        p24_hx_calib = ''  # !! need to add

        # two records for 4/20
        p24_E4 = [
            '230418-145210',
            '230419-130807',
            '230420-145444',
            '230420-224259',
            '230421-132148',
            '230422-141833',
            '230425-161624',
            '230426-131230',
            '230427-130441',
            '230428-131846',
            '230429-152039',
        ]

        p24_hx = [
            '265002',
            '265068',
            '265176',
            '265178',
            '265221',
            '265262',
            '265475',
            '265558',
            '265592',
            '265668',
            '265680',
        ]

        p24_redcap_avail = []  # !! need to add

        p24 = {
            'status': status[24],
            'e4sn': 'A037F7',
            'hxsn': '8536',
            'complete days': partcipant_days[24],
            'RedCap available': p24_redcap_avail,
            'calibration': [p24_E4_calib, p24_hx_calib],
            'files': [p24_E4, p24_hx],
        }

        # p29

        p29_E4_calib = ''  # !! need to add
        p29_hx_calib = ''  # !! need to add

        p29_E4 = [
            '230418-163352',
            '230419-135409',
            '230420-135655',
            '230421-140738',
            '230422-170513',
            '230425-164136',
            '230426-192638',
            '230427-135219',
            '230428-163147',
            '230429-171324',
        ]

        p29_hx = [
            '264989',
            '265066',
            '265159',
            '265222',
            ['265259', '265260', '265263'],  # there were three files for the hex
            '265476',
            '265564',
            '265598',
            ['265671', '265672'],  # there were two files for the hex
            '265681',
        ]

        p29_redcap_avail = []  # !! need to add

        p29 = {
            'status': status[29],
            'e4sn': 'A03ADB',
            'hxsn': '8872',
            'complete days': partcipant_days[29],
            'RedCap available': p29_redcap_avail,
            'calibration': [p29_E4_calib, p29_hx_calib],
            'files': [p29_E4, p29_hx],
        }

        # p33

        p33_E4_calib = ''  # !! need to add
        p33_hx_calib = ''  # !! need to add

        p33_E4 = []  # !! need to add

        p33_hx = []  # !! need to add

        p33_redcap_avail = []  # !! need to add

        p33 = {
            'status': status[33],
            'e4sn': 'A03ADB',
            'hxsn': '7234',
            'complete days': partcipant_days[33],
            'RedCap available': p33_redcap_avail,
            'calibration': [p33_E4_calib, p33_hx_calib],
            'files': [p33_E4, p33_hx],
        }

        # p39

        p39_E4_calib = ''  # !! need to add
        p39_hx_calib = ''  # !! need to add

        p39_E4 = [
            '230829-125305',
            '230830-142221',
            '230831-133309',
            '230901-112545',
            '230902-143059',
            '230905-115307',
            '230906-135127',
            '230907-125110',
            '230908-154904',
            '230909-121113',
        ]

        p39_hx = [
            '272539',
            '272631',
            '272766',
            '272812',
            '272824',
            ['272935', '272936', '272937'],
            '273036',
            '273163',
            '273244',
            '273256',
        ]

        p39_redcap_avail = []  # !! need to add

        p39 = {
            'status': status[39],
            'e4sn': 'A036EE',
            'hxsn': '8872',
            'complete days': partcipant_days[39],
            'RedCap available': p39_redcap_avail,
            'calibration': [p39_E4_calib, p39_hx_calib],
            'files': [p39_E4, p39_hx],
        }

        # p92

        p92_E4_calib = ''  # !! need to add
        p92_hx_calib = ''  # !! need to add

        p92_E4 = [
            '230726-163005',
            '230727-145538',
            '230728-124335',
            '230729-142926',
            '230730-143129',
            '230801-124652',
            '230802-133129',
            '230803-120906',
        ]

        p92_hx = [
            '270767',
            '270830',
            '270889',
            '270896',
            '270919',
            '271137',
            '271228',
            '271303',
        ]

        p92_redcap_avail = []  # !! need to add

        p92 = {
            'status': status[92],
            'e4sn': 'A04BA8',
            'hxsn': '8872',
            'complete days': partcipant_days[92],
            'RedCap available': p92_redcap_avail,
            'calibration': [p92_E4_calib, p92_hx_calib],
            'files': [p92_E4, p92_hx],
        }

        # p216

        p216_E4_calib = ''  # !! need to add
        p216_hx_calib = ''  # !! need to add

        # Two 09/21 days
        p216_E4 = [
            '230912-165740',
            '230913-174307',
            '230914-163147',
            '230915-190823',
            '230916-182941',
            '230919-170330',
            '230920-201656',
            '230921-164156',
            '230921-214002',
            '230922-143405',
            '230923-180623',
        ]
        p216_hx = [
            '273357',
            '273435',
            '273530',
            '273560',
            '273570',
            '273666',
            '273833',
            '273834',
            '273835',
            '273882',
            '273889',
        ]

        p216_redcap_avail = []  # !! need to add

        p216 = {
            'status': status[216],
            'e4sn': 'A036EE',
            'hxsn': '44675',
            'complete days': partcipant_days[216],
            'RedCap available': p216_redcap_avail,
            'calibration': [p216_E4_calib, p216_hx_calib],
            'files': [p216_E4, p216_hx],
        }

        # p221

        p221_E4_calib = ''  # !! need to add
        p221_hx_calib = ''  # !! need to add

        p221_E4 = []  # !! need to add
        p221_hx = []  # !! need to add

        p221_redcap_avail = []  # !! need to add

        p221 = {
            'status': status[221],
            'e4sn': 'A0343F',
            'hxsn': '8783',
            'complete days': partcipant_days[221],
            'RedCap available': p221_redcap_avail,
            'calibration': [p221_E4_calib, p221_hx_calib],
            'files': [p221_E4, p221_hx],
        }

        fs = {
            'ECG': 256,
            'BVP': 64,
            'BR': 1,
            'TEMP': 4,
            'EDA': 4,
            'ACC_hx': 64,
            'ACC_e4': 32,
        }
        all_p_metadata = {
            'list of participant IDs': participants,
            1: p1,
            5: p5,
            8: p8,
            9: p9,
            13: p13,
            15: p15,
            24: p24,
            29: p29,
            33: p33,
            39: p39,
            92: p92,
            216: p216,
            221: p221,
            'fs': fs,
        }

        with open(wear_path + 'all_p_metadata_wear.json', 'w', encoding='utf-8') as fp:
            json.dump(all_p_metadata, fp)
        print('metadata file created')


if __name__ == '__main__':
    # do this:
    sys.argv
    create_radwear_metadata()
    create_wear_metadata()
