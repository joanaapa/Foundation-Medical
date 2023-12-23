from utils import *
from .bases import BaseSet
from scipy.io import mmread
from torchvision.transforms import ToTensor, ToPILImage


DATA_INFO = {
              "DDSM": {"dataset_location": "DDSM"},
              "CheXpert": {"dataset_location": "CheXpert"},
              "ISIC2019": {"dataset_location": "ISIC2019"},
              "APTOS2019": {"dataset_location": "APTOS2019"},
              "Camelyon": {"dataset_location": "Camelyon"},    
}


class CheXpert(BaseSet):

    img_channels = 1
    is_multiclass = False
    task = 'classification'    
    mean = (0.503,)
    std = (0.292,)
    knn_nhood = 200
    target_metric = 'roc_auc'
    int_to_labels = {0: 'No Finding',
                     1: 'Enlarged Cardiomediastinum',
                     2: 'Cardiomegaly',
                     3: 'Lung Opacity',
                     4: 'Lung Lesion',
                     5: 'Edema',
                     6: 'Consolidation',
                     7: 'Pneumonia',
                     8: 'Atelectasis',
                     9: 'Pneumothorax',
                     10: 'Pleural Effusion',
                     11: 'Pleural Other',
                     12: 'Fracture',
                     13: 'Support Devices'
                     }
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["CheXpert"]["dataset_location"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_dataset()
        self.transform, self.resizing = self.get_transforms()
    
    def get_data_as_list(self, data_loc):
        data_list = []     
        testval_size = 0.2
        datainfo = pd.read_csv(data_loc, engine='python')
        datainfo = datainfo.sort_values(by=['Path'])
        val_id_json = os.path.join(self.root_dir, 'val_ids.json') 
        testval_size = int(len(datainfo) * testval_size) + 15  # the 15 is just to make a better patient split 
        
        # data = np.array(datainfo.iloc[:, -self.n_classes:].fillna(0).replace(-1, 0).values.tolist())
        if self.mode == 'train':
            data = datainfo.loc[testval_size:]
        else:
            data = datainfo.loc[:testval_size]
            if self.mode in ['val', 'eval']:
                data = data.loc[:int(len(data)/2)]
            elif self.mode in ['test']:
                data = data.loc[int(len(data)/2):]                
            else:
                raise ValueError(f"mode {self.mode} not understood")
        
        # we use the U-Zeroes model i.e. we replace NaNs and -1s with 0s
        labels = data.iloc[:, -self.n_classes:].fillna(0).replace(-1, 0).values.tolist()
        # converting to 0: 'No Finding' every label that has only zeros
        for l in range(len(labels)):
            if sum(labels[l]) == 0:
                labels[l][self.labels_to_int['No Finding']] = 1
   
        # remove multilabels and keep only single labels (the 0 everywhere is also omitted)
        if self.is_multiclass:
            labels = [label for label in labels if sum(label) == 1]
            labels = np.where(np.array(labels)==1)[1].tolist()
        img_paths = data['Path'].values.tolist()
        img_paths = [os.path.join(self.root_dir, *img_path.split('/')[1:]) for img_path in img_paths]
        for img_path, label in zip(img_paths, labels):
            data_list.append({'img_path': img_path, 'label': label, 'dataset': self.name})
                    
        return data_list
    
    def get_dataset(self):
        if self.mode in ['train', 'val', 'eval', 'test']:
            self.df_path = os.path.join(self.root_dir, 'train.csv')
        else:
            self.df_path = os.path.join(self.root_dir, 'valid.csv')
        return self.get_data_as_list(self.df_path) 
    
    
class DDSM(BaseSet):
    img_channels = 1
    is_multiclass = True
    task = 'classification'   
    knn_nhood = 200
    target_metric = 'roc_auc'
    
    def __init__(self, dataset_params, mode='train', n_class=2, is_patch=False):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.n_class = n_class
        self.is_patch = is_patch
        self.export_labels_as_int()
        self.init_stats()
        self.n_classes = len(self.int_to_labels)
        assert self.n_classes == self.n_class
        self.dataset_location = DATA_INFO["DDSM"]["dataset_location"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.label_mode = '{}class'.format(self.n_class)
        
        self.data = self.get_dataset()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self, data_loc):
        data_list = []
        data = pd.read_csv(data_loc, sep=" ", header=None, engine='python')
        if self.is_patch:
            data.columns = ["img_path", "label"]
            for img_path, label in zip(data['img_path'], data['label']):
                img_path = os.path.join(*img_path.split("/")[1:])
                img_path = os.path.join(self.root_dir, 'ddsm_patches', img_path)
                data_list.append({'img_path': img_path, 'label': label, 'dataset': self.name})
        else:
            data.columns = ["img_path"]
            txt_to_lbl = {'normal': 0, 'benign': 1, 'cancer': 2}
            for img_path in data['img_path']:
                img_path = os.path.join(self.root_dir, 'ddsm_raw', img_path)
                label = os.path.basename(img_path).split("_")[0]
                label = txt_to_lbl[label]
                if self.n_classes == 2 and label > 1:
                    label = 1
                if not self.is_multiclass:
                    label = [float(label)]
                data_list.append({'img_path': img_path, 'label': label, 'dataset': self.name})
                    
        return data_list
    
    def get_dataset(self):
        if self.is_patch:
            self.df_path = os.path.join(self.root_dir, 'ddsm_labels', self.label_mode)
        else:
            self.df_path = os.path.join(self.root_dir, 'ddsm_raw_image_lists')
        if self.mode == 'train':
            self.df_path = os.path.join(self.df_path, 'train.txt')
        elif self.mode in ['val', 'eval']:
            self.df_path = os.path.join(self.df_path, 'val.txt')
        elif self.mode == 'test':
            self.df_path = os.path.join(self.df_path, 'test.txt')
        return self.get_data_as_list(self.df_path)
            
    def init_stats(self):
        if self.is_patch:
            self.mean = (0.44,)
            self.std = (0.25,)
        else:
            self.mean = (0.286,)
            self.std = (0.267,)      
        
    def export_labels_as_int(self):
        if self.n_class == 3:
            self.int_to_labels = {
                0: 'Normal',
                1: 'Benign',
                2: 'Cancer'
            }
        else:
            self.int_to_labels = {
                0: 'Normal',
                1: 'Cancer'
            }
        self.labels_to_int = {val: key for key, val in self.int_to_labels.items()} 
        
    
class ISIC2019(BaseSet):
    
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = [0.66776717, 0.52960888, 0.52434725]
    std = [0.22381877, 0.20363036, 0.21538623]
    knn_nhood = 200    
    int_to_labels = {
        0: 'Melanoma',
        1: 'Melanocytic nevus',
        2: 'Basal cell carcinoma',
        3: 'Actinic keratosis',
        4: 'Benign keratosis',
        5: 'Dermatofibroma',
        6: 'Vascular lesion',
        7: 'Squamous cell carcinoma'
    }
    target_metric = 'recall'
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["ISIC2019"]["dataset_location"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        datainfo = pd.read_csv(os.path.join(self.root_dir, 'ISIC_2019_Training_GroundTruth.csv'), engine='python')
        metadata = pd.read_csv(os.path.join(self.root_dir, 'ISIC_2019_Training_Metadata.csv'), engine='python')
        labellist = datainfo.values[:, 1:].nonzero()[1].tolist()
        img_names = datainfo.values[:, 0].tolist()
        img_names = [os.path.join(self.root_dir, 'train',  imname + '.jpg') for imname in img_names]
        dataframe = pd.DataFrame(list(zip(img_names, labellist)), 
                                 columns=['img_path', 'label'])
        
        val_id_json = os.path.join(self.root_dir, 'val_ids.json')
        train_ids, test_val_ids = self.get_validation_ids(total_size=len(dataframe), val_size=0.2, 
                                                          json_path=val_id_json, 
                                                          dataset_name=self.name)
        val_ids = test_val_ids[:int(len(test_val_ids)/2)]
        test_ids = test_val_ids[int(len(test_val_ids)/2):]     
        
        if self.mode == 'train':
            data = dataframe.loc[train_ids, :]
        elif self.mode in ['val', 'eval']:
            data = dataframe.loc[val_ids, :]
        else:
            data = dataframe.loc[test_ids, :]
        labels = data['label'].values.tolist()
        img_paths = data['img_path'].values.tolist()
        data_list = [{'img_path': img_path, 'label': label, 'dataset': self.name}
                     for img_path, label in zip(img_paths, labels)]
                    
        return data_list  
    
    
class APTOS2019(BaseSet):
    
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = (0.415, 0.221, 0.073)
    std = (0.275, 0.150, 0.081)
    int_to_labels = {
        0: 'No DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferative DR'
    }
    target_metric = 'quadratic_kappa'
    knn_nhood = 200    
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["APTOS2019"]["dataset_location"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        datainfo = pd.read_csv(os.path.join(self.root_dir, 'train.csv'), engine='python')
        labellist = datainfo.diagnosis.tolist()
        img_names = datainfo.id_code.tolist()
        img_names = [os.path.join(self.root_dir, 'train_images', imname + '.png') for imname in img_names]
        dataframe = pd.DataFrame(list(zip(img_names, labellist)), 
                                 columns=['img_path', 'label'])
        
        val_id_json = os.path.join(self.root_dir, 'val_ids.json')
        train_ids, test_val_ids = self.get_validation_ids(total_size=len(dataframe), val_size=0.3, 
                                                          json_path=val_id_json, 
                                                          dataset_name=self.name)
        val_ids = test_val_ids[:int(len(test_val_ids)/2)]
        test_ids = test_val_ids[int(len(test_val_ids)/2):]     
        
        if self.mode == 'train':
            data = dataframe.loc[train_ids, :]
        elif self.mode in ['val', 'eval']:
            data = dataframe.loc[val_ids, :]
        else:
            data = dataframe.loc[test_ids, :]
        labels = data['label'].values.tolist()
        img_paths = data['img_path'].values.tolist()
        data_list = [{'img_path': img_path, 'label': label, 'dataset': self.name}
                     for img_path, label in zip(img_paths, labels)]
                    
        return data_list    
    
    
class Camelyon(BaseSet):
    
    img_channels = 3
    is_multiclass = True
    task = 'classification'    
    knn_nhood = 200    
    mean = [0.69991202, 0.53839318, 0.69108667]
    std = [0.2308955 , 0.27435492, 0.20865005]
    int_to_labels = {0: 'Normal',
                    1: 'Tumor'}   
    target_metric = 'roc_auc'    
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["Camelyon"]["dataset_location"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
            
    def get_data_as_list(self):
        data_list = []  
        prefix = "camelyonpatch_level_2_split_"
        if self.mode == "train":
            mod = "train"
        elif self.mode in ["val", "eval"]:
            mod = "valid"
        elif self.mode == "test":
            mod = "test"
        else:
            raise ValueError(f"Mode {self.mode} not understood")
                    
        h5_data = os.path.join(self.root_dir, f"{prefix}{mod}_x.h5")
        h5_labels = os.path.join(self.root_dir, f"{prefix}{mod}_y.h5")  
        meta = pd.read_csv(os.path.join(self.root_dir, f"{prefix}{mod}_meta.csv"), engine='python')
        png_data_path = os.path.join(self.root_dir, f"{mod}_png_files")
        labels_data_path = os.path.join(self.root_dir, f"{mod}_labels.json") 
        check_dir(png_data_path)
        
        img_paths = [os.path.join(png_data_path, f"img_{img_i}.png") for img_i in range(len(meta))]
        
        data = None
        labels = None
        if os.path.exists(h5_data):
            data = h5py.File(h5_data,'r')
        if os.path.exists(h5_labels):
            labels = h5py.File(h5_labels,'r')['y']
        
        # getting the labels
        if os.path.exists(labels_data_path) and (len(load_json(labels_data_path)) == len(meta)):
            labels = load_json(labels_data_path)
        else:
            if is_rank0(torch.cuda.current_device()):
                print("Extracting labels from h5")
                labels = [int(lbl.flatten()[0]) for lbl in h5py.File(h5_labels,'r')['y']]
                save_json(labels, labels_data_path)
            synchronize()
            labels = load_json(labels_data_path)
        
        # getting the data
        if sum([os.path.exists(p) for p in img_paths]) == len(meta):
            pass
        elif isinstance(data, h5py._hl.files.File):
            if is_rank0(torch.cuda.current_device()):
                data = data['x']                
                def deserialize(img, img_path, image_size=(512,512),
                                interpolation_method=Image.LANCZOS, resize=False):
                    if os.path.exists(img_path):
                        return
                    img = Image.fromarray(img)
                    if resize:
                        img = img.resize(image_size, resample=interpolation_method)
                    img.save(img_path)
                print("Converting h5 container to PNG data.. This might take a while")
                for idx in tqdm(range(len(data))):
                    deserialize(data[idx], img_paths[idx])
            synchronize()
            return self.get_data_as_list()
        else:
            raise ValueError("Data: empty container or incorect path")            
            
        data_list = [{'img_path':img_path, 'label':label, 'dataset':self.name} 
                     for img_path, label in zip(img_paths, labels)]
                    
        return data_list      
    
