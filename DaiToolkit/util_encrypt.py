import yaml
from cryptography.fernet import Fernet

from . import util_readfile

# key from Fernet.generate_key()
CRYPTOGRAPHY_KEY = util_readfile.read_yaml_rel_path('crypto.yaml')['key']
CRYPTOOBJ = Fernet(CRYPTOGRAPHY_KEY)


def encrypt_yaml(file_path, flds_need_encrypt):
    if isinstance(flds_need_encrypt, str):
        flds_need_encrypt = [flds_need_encrypt]

    with open(file_path) as f:
        yaml_config = yaml.safe_load(f)

    for fld in flds_need_encrypt:
        if fld not in list(yaml_config.keys()):
            raise Exception('Error! [' + fld + '] not in yaml file')
        else:
            yaml_config[fld] = CRYPTOOBJ.encrypt(yaml_config[fld])

    with open(file_path, 'w') as f:
        yaml.dump(yaml_config, f)


def decrypt_yaml(file_path, flds_need_decrypt):
    if isinstance(flds_need_decrypt, str):
        flds_need_decrypt = [flds_need_decrypt]

    with open(file_path) as f:
        yaml_config = yaml.safe_load(f)

    for fld in flds_need_decrypt:
        if fld not in list(yaml_config.keys()):
            raise Exception('Error! [' + fld + '] not in yaml file')
        else:
            yaml_config[fld] = CRYPTOOBJ.decrypt(yaml_config[fld])
    return yaml_config
