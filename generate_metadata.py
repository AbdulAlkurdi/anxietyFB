import process_redcap
import json
import sys
from pathlib import Path

def create_metadata (radwear_path = '/mnt/c/Users/alkurdi/Desktop/Vansh/data/RADWear/', force_update = False): 
    redcap_path = radwear_path+'REDCap responses/'

    my_file = Path(radwear_path+'all_p_metadata.json')
    if my_file.is_file() and not force_update:
        print('metadata file exists')
        with open(radwear_path+'all_p_metadata.json', 'rb') as f:
            all_p_metadata = json.load(f)
            print('json loaded')
    else: 
        if force_update:
            print('metadata data exists but update forced')
        else:
            print('metadata data is not ready. will be created')
    
        
        participants = [4, 7, 9, 12, 14, 16, 17, 18, 21] #dropped 5, 20

        status = {4:'calibration only', 
            5:'disregard, missing many E4 + noisy', 
            7:'complete', 
            9:'complete', 
            12:'complete', 
            14:'complete',
            16:'ongoing',
            17:'complete', 
            18:'calibration only', 
            20:'disregard, calibration only and missing e4', 
            21:'ongoing'}
        
        '''
        keys for data:
        low anxiety: LA
        high anxiety: HA
        e4sn: e4 serial number
        hxsn: hexoskin serial number
        '''


        # label definitions
        calib_dict = {'meditation': 0, 'cpt': 1}
        rot_anx_dict = {'calibration': 0, 'LA': 1, 'HA': 2}

        # participant days: participant_id: [number of days for low anxiety, number of days for high anxiety]
        partcipant_days = {4:[0,0], 5:[0,8], 7:[11,10], 9:[10,10], 12:[8,10], 14:[9,10], 16:[9,0], 17:[10,10], 18:[0,0], 20:[0,0], 21:[9,0]}

        total = 0
        for each in list(partcipant_days.values()):
            total += each[0]
            total += each[1]
        print('number of days total of redcap data available for RADWear in the wild: ',total)


        # p4
        p4_E4_calib = '220714-232953'
        p4_hx_calib = '248835'
        
        p4_E4_LA = []
        p4_hx_LA = []
        p4_E4_HA = []
        p4_hx_HA = []

        p4_redcap_avail = [[],[]]
        p4 = {'status':status[4],
            'e4sn':'A0381C',
            'hxsn':'4772',
            'complete days':partcipant_days[4],
            'RedCap available':p4_redcap_avail,
            'calibration':[p4_E4_calib,p4_hx_calib],
            'LA':[p4_E4_LA,p4_hx_LA],
            'HA':[p4_E4_HA,p4_hx_HA]}
        
        # p5

        p5_E4_calib = '220704-164707'
        p5_hx_calib = '250343'

        p5_E4_LA = []
        p5_hx_LA = []
        p5_E4_HA = ['220927-115055', '220928-115104', '220929-114633', '220930-114429',
                    '221001-120807', '221003-114642', '221004-120334', '221005-110525' + '221005-145538']
        p5_hx_HA = ['253647', '253648', '253649', '253650',
                    '253651', '253652', '253753', '254429']
        p5_redcap_avail = [[],[1,1,1,1,
                            1,1,1,0]]

        #p5_redcap = [all_participants_redcap_dict[5]] # not done yet

        p5 = {'status':status[5],
            'e4sn':'A037F7', 
            'hxsn':'8783',
            'complete days':partcipant_days[5],
            'RedCap available':p5_redcap_avail,
            'calibration':[p5_E4_calib,p5_hx_calib],
            'LA':[p5_E4_LA,p5_hx_LA],
            'HA':[p5_E4_HA,p5_hx_HA]}


        # p7


        p7_E4_calib = '220712-181114'
        p7_hx_calib = '248582'

        p7_E4_LA = ['220829-121320', '220830-114442', '220831-120141', '220901-113936', '220902-114700', '220906-113947',
                    '220907-120003', '220908-120451', '220909-120309', '220912-151726', '220913-134458']
        p7_hx_LA = [0,'251292', '251331', '251374', '251399', '251559',
                    '251577', '251622', '251779', '251928', '251984']

        p7_E4_HA = ['230410-115059', '230411-122423', '230412-121912', '230413-123640', '230414-122016',
                    0, '230418-123119', '230419-121005', '230420-144405', '230421-163430' ]
        p7_hx_HA = ['264721', '264747', '264775', '264809', '264836',
                    '265005', '265006', '265069', '265174', '265224']

        p7_redcap_avail = [[0,1,1,1,1,1,
                        1,1,1,1,1],
                         [1,1,1,1,1,
                          1,1,1,1,1]] # ?, ?]
        #p7_redcap = [all_participants_redcap_dict[7]] # not done yet

        p7 = {'status':status[7],
            'e4sn':'A04C05', 
            'hxsn':'8550',
            'complete days':partcipant_days[7],
            'RedCap available':p7_redcap_avail,
            'calibration':[p7_E4_calib,p7_hx_calib],
            'LA':[p7_E4_LA,p7_hx_LA],
            'HA':[p7_E4_HA,p7_hx_HA]}
        
        # p9

        p9_E4_calib = '220704-164707'
        p9_hx_calib = '248258'

        p9_E4_LA = ['220801-180621', '220802-163608', '220803-145947', '220804-151805', '220805-134635',
                    '220808-145300', '220809-142505', '220810-152951', '220811-161838', '220812-154403']
        p9_hx_LA = ['249726', '249805', '249918', '249974', '250026',
                    '250069', '250118', '250325', '250410', '250464']
        p9_E4_HA = ['220815-122425', '220816-114335', '220817-113455', '220818-120324', '220819-114116',
                    '220821-214544', '220822-223419', '220823-222539', '220824-230220', '220825-232501'+'220826-071128']
        p9_hx_HA = ['250527', '250598', '250774', '250850', '250875',
                    '250947', '250977', '251011', '251094', '251136']

        p9_redcap_avail = [[1,1,1,1,1,
                        1,1,1,1,1],
                        [1,1,1,1,1,
                        1,1,1,1,0]] # not done yet
        #p9_redcap = [all_participants_redcap_dict[9]]

        p9 = {'status':status[9],
            'e4sn':'A04BA8',
            'hxsn':'8872',
            'complete days':partcipant_days[9],
            'RedCap available':p9_redcap_avail,
            'calibration':[p9_E4_calib,p9_hx_calib],
            'LA':[p9_E4_LA,p9_hx_LA],
            'HA':[p9_E4_HA,p9_hx_HA]}
        
        # p12

        p12_E4_calib = '220701-002248'
        p12_hx_calib = '248092'

        p12_E4_LA = ['220718-132014', '220719-130314', '220720-120906', '220721-122543',
                    '220725-121303', '220726-123528', '220727-121705', '220801-120508']
        p12_hx_LA = ['248962', '249050', '249152', '249272',
                    '249373', '249417', '249468', '251204']
        p12_E4_HA = ['220911-220003', '220912-222451', '220913-222507', '220915-221059', '220916-060310',
                    '220919-112244', '220920-112551', '220921-122019', '220922-113528', '220923-113640']
        p12_hx_HA = ['251915', '251983', '252042', '252251', '252252',
                    '252315', '252395', '252522', '252609', '252733']

        p12_redcap_avail = [[1,1,1,1,
                            1,1,1,1],
                            [1,1,1,1,1,
                            1,1,1,1,1]] #redcap data is shifted and dates aren't all correct
        #p12_redcap = [all_participants_redcap_dict[12]]

        p12 = {'status':status[12],
                'e4sn':'A038E2',
                'hxsn':'7234',
                'complete days':partcipant_days[12],
                'RedCap available':p12_redcap_avail,
                'calibration':[p12_E4_calib,p12_hx_calib],
                'LA':[p12_E4_LA,p12_hx_LA],
                'HA':[p12_E4_HA,p12_hx_HA]}
        
        # p14

        p14_E4_calib = '220620-224412'
        p14_hx_calib = '247555'

        p14_E4_LA = ['220718-115638', '220719-160849', '220720-163536', '220721-125931', '220722-142715',
                    '220726-133230', 0, '220728-121347', '220729-112349']
        p14_hx_LA = ['248966', '249044', '249151', '249201', '249278',
                    '249406', '249469', '249524', '249561']
        p14_E4_HA = ['230501-113540', '230502-142834', '230503-115351', '230504-115657', '230505-115115',
                    '230508-113742', '230509-133128', '230510-122315', '230511-125537', '230512-114245']
        p14_hx_HA = [0, '265813', '265915', '266004', '266106',
                    '266187', '266302', '266365', '266415', '266815']

        p14_redcap_avail = [[1,1,1,1,1,
                            1,1,1,1],
                            [1,1,1,0,0,
                            1,0,1,1,1]]
        #p14_redcap = [all_participants_redcap_dict[14]]

        p14 = {'status':status[14],
                'e4sn':'A036EE',
                'hxsn':'44675',
                'complete days':partcipant_days[14],
                'RedCap available':p14_redcap_avail,
                'calibration':[p14_E4_calib,p14_hx_calib],
                'LA':[p14_E4_LA,p14_hx_LA],
                'HA':[p14_E4_HA,p14_hx_HA]} 
        
        # p16

        p16_E4_calib = '230727-000346'
        p16_hx_calib = '270761'

        p16_E4_LA = ['230828-130409', '230829-131230', '230830-130830', '230831-124044', '230901-124432',
                    '230905-132521', '230906-131815', '230907-125434', '230908-131813']
        p16_hx_LA = ['272490', '272532', '272876', '272877', '272878',
                    '272931', '273035', '273155', '273236']
        p16_E4_HA = []
        p16_hx_HA = []

        p16_redcap_avail = [[1,1,1,1,1,
                            1,1,1,1],
                            []]
        #p16_redcap = [all_participants_redcap_dict[16]]

        p16 = {'status':status[16],
                'e4sn':'A0343F',
                'hxsn':'8783',
                'complete days':partcipant_days[16],
                'RedCap available':p16_redcap_avail,
                'calibration':[p16_E4_calib,p16_hx_calib],
                'LA':[p16_E4_LA,p16_hx_LA],
                'HA':[p16_E4_HA,p16_hx_HA]}
        
        # p17

        p17_E4_calib = '230727-000250'
        p17_hx_calib = '270766'

        p17_E4_LA = ['230807-125842', '230808-132814', '230809-134843', '230810-131337', '230811-104045',
                    '230814-102700', '230815-113210', '230816-110843', 0, '230818-112608']
        p17_hx_LA = ['271384', '271418', '271455', '271493', '271514',
                    '271596', '271644', '271739', '271784', '271833']
        p17_E4_HA = ['230821-104414', '230822-133555', '230823-105017', '230824-102510', '230825-113427',
                    '230828-114102', '230829-220819', '230830-101242', '230831-101208', '230901-113018']
        p17_hx_HA = ['272004', '272113', '272204', '272304', '272422',
                    '272489', '272536', '272647', '272730', '272804']

        p17_redcap_avail = [[1,1,1,1,1,
                            1,1,1,1,1],
                            [1,1,1,1,1,
                            1,1,1,1,1]]
        #p17_redcap = [all_participants_redcap_dict[17]]

        p17 = {'status':status[17],
                'e4sn':'A0340C',
                'hxsn':'8536',
                'complete days':partcipant_days[17],
                'RedCap available':p17_redcap_avail,
                'calibration':[p17_E4_calib,p17_hx_calib],
                'LA':[p17_E4_LA,p17_hx_LA],
                'HA':[p17_E4_HA,p17_hx_HA]}
        
        # p18

        p18_E4_calib = '230727-000306'
        p18_hx_calib = '270759'

        p18_E4_LA = []
        p18_hx_LA = []
        p18_E4_HA = []
        p18_hx_HA = []

        #p18_redcap_avail = [all_participants_redcap_dict[18]]
        p18_redcap_avail = [[],[]]

        p18 = {'status':status[18],
                'e4sn':'A038E2',
                'hxsn':'',
                'complete days':partcipant_days[18],
                'RedCap available':p18_redcap_avail,
                'calibration':[p18_E4_calib,p18_hx_calib],
                'LA':[p18_E4_LA,p18_hx_LA],
                'HA':[p18_E4_HA,p18_hx_HA]}        
        # p20

        p20_E4_calib = ''
        p20_hx_calib = '270760'

        p20_E4_LA = []
        p20_hx_LA = []
        p20_E4_HA = []
        p20_hx_HA = []

        #p20_redcap_avail = [all_participants_redcap_dict[20]]
        p20_redcap_avail = []     

        p20 = {'status':status[20],
                'e4sn':'',
                'hxsn':'',
                'complete days':partcipant_days[20],
                'RedCap available':p20_redcap_avail,
                'calibration':[p20_E4_calib,p20_hx_calib],
                'LA':[p20_E4_LA,p20_hx_LA],
                'HA':[p20_E4_HA,p20_hx_HA]}
        
        # p21

        p21_E4_calib = '230727-000318'
        p21_hx_calib = '270764'

        p21_E4_LA = ['230821-124927', '230822-160632', '230823-120957', '230825-124152',
                    '230828-114504', '230829-114425', '230830-114324', '230831-114743', '230901-120509', '230907-170014']
        p21_hx_LA = ['272105', '272124', '272189', '272424',
                    '272482', '272535', '272628', '272765', '272806', '273162']
        p21_E4_HA = []
        p21_hx_HA = []

        p21_redcap_avail = [[1,1,1,1,1,
                            1,1,1,1,1],
                            []]
        #p21_redcap = [all_participants_redcap_dict[21]]

        p21 = {'status':status[21],
                'e4sn':'A04C05',
                'hxsn':'8550',
                'complete days':partcipant_days[21],
                'RedCap available':p21_redcap_avail,
                'calibration':[p21_E4_calib,p21_hx_calib],
                'LA':[p21_E4_LA,p21_hx_LA],
                'HA':[p21_E4_HA,p21_hx_HA]}
        fs = {'ECG': 256, 'BVP' : 64, 'BR' : 1, 'TEMP': 4, 'EDA' : 4, 'ACC_hx' : 64, 'ACC_e4' : 32}
        all_p_metadata = {'list of participant IDs':participants, 4:p4, 5:p5, 7:p7, 9:p9, 12:p12, 14:p14, 16:p16, 17:p17, 18:p18, 20:p20, 21:p21, 'fs':fs}

        with open(radwear_path+'all_p_metadata.json', 'w') as fp:
            json.dump(all_p_metadata, fp)
        print('metadata file created')


if __name__ == '__main__':
    
    #do this:
    sys.argv
    create_metadata()