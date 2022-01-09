import os
import hashlib
import random

def get_md5_of_file(filename):
    """
    get md5 of a file
    :param filename:
    :return:
    """
    if not os.path.isfile(filename):
        return None
    myhash = hashlib.md5()
    with open(filename, 'rb') as f:
        while True:
            b = f.read()
            if not b:
                break
            myhash.update(b)
    return myhash.hexdigest()


def get_md5_of_folder(dir_):
    """
    get md5 of a folder
    :param dir_:
    :return:
    """
    if not os.path.isdir(dir_):
        return None
    MD5File = "{}_tmp.md5".format(str(random.randint(0, 1000)).zfill(6))
    with open(MD5File, 'w') as outfile:
        for root, subdirs, files in os.walk(dir_):
            for file in files:
                filefullpath = os.path.join(root, file)
                md5 = get_md5_of_file(filefullpath)
                outfile.write(md5)
    val = get_md5_of_file(MD5File)
    os.remove(MD5File)
    return val