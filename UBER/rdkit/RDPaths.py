import os
# unset so to trigger exceptions and track use: RDBaseDir=os.environ['RDBASE']
RDCodeDir=os.path.join(r'D:\bld\rdkit_1657061356439\_h_env\Lib\site-packages','rdkit')
# not really hard-coded alternative RDCodeDir=os.path.dirname(__file__)
_share = os.path.join(r'C:/Users/theco/anaconda3/envs/deeplearning/Library', r'share/RDKit')
RDDataDir=os.path.join(_share,'Data')
RDDocsDir=os.path.join(_share,'Docs')
RDProjDir=os.path.join(_share,'Projects')
RDContribDir=os.path.join(_share,'Contrib')
