"""Frozen benchmark roster, copied from ruadapt/CaQR main benchmarks directory 2026-11-11.

CaQR main GitHub tree and atenaraeisi/GidNET-Qubit-Reuse-Algorithm experiments
benchmarks_compact each contain the same 168 named QASM files. Excluding five
`test*.qasm` and one `_example.qasm` leaves 162 named benchmark circuits.
"""
NAMES = '''
0410184_169.qasm
3_17_13.qasm
4_49_16.qasm
4gt10-v1_81.qasm
4gt11_82.qasm
4gt11_83.qasm
4gt11_84.qasm
4gt12-v0_86.qasm
4gt12-v0_87.qasm
4gt12-v0_88.qasm
4gt12-v1_89.qasm
4gt13-v1_93.qasm
4gt13_90.qasm
4gt13_91.qasm
4gt13_92.qasm
4gt4-v0_72.qasm
4gt4-v0_73.qasm
4gt4-v0_78.qasm
4gt4-v0_79.qasm
4gt4-v0_80.qasm
4gt4-v1_74.qasm
4gt5_75.qasm
4gt5_76.qasm
4gt5_77.qasm
4mod5-bdd_287.qasm
4mod5-v0_18.qasm
4mod5-v0_19.qasm
4mod5-v0_20.qasm
4mod5-v1_22.qasm
4mod5-v1_23.qasm
4mod5-v1_24.qasm
4mod7-v0_94.qasm
4mod7-v1_96.qasm
9symml_195.qasm
C17_204.qasm
adr4_197.qasm
aj-e11_165.qasm
alu-bdd_288.qasm
alu-v0_26.qasm
alu-v0_27.qasm
alu-v0_27_example.qasm
alu-v1_28.qasm
alu-v1_29.qasm
alu-v2_30.qasm
alu-v2_31.qasm
alu-v2_32.qasm
alu-v2_33.qasm
alu-v3_34.qasm
alu-v3_35.qasm
alu-v4_36.qasm
alu-v4_37.qasm
bv_n10.qasm
cc_n10.qasm
clip_206.qasm
cm152a_212.qasm
cm42a_207.qasm
cm82a_208.qasm
cm85a_209.qasm
cnt3-5_179.qasm
cnt3-5_180.qasm
co14_215.qasm
con1_216.qasm
cycle10_2_110.qasm
dc1_220.qasm
dc2_222.qasm
decod24-bdd_294.qasm
decod24-enable_126.qasm
decod24-v0_38.qasm
decod24-v1_41.qasm
decod24-v2_43.qasm
decod24-v3_45.qasm
dist_223.qasm
ex-1_166.qasm
ex1_226.qasm
ex2_227.qasm
ex3_229.qasm
f2_232.qasm
graycode6_47.qasm
ground_state_estimation_10.qasm
ham15_107.qasm
ham3_102.qasm
ham7_104.qasm
hwb4_49.qasm
hwb5_53.qasm
hwb6_56.qasm
hwb7_59.qasm
hwb8_113.qasm
hwb9_119.qasm
inc_237.qasm
ising_model_10.qasm
ising_model_13.qasm
ising_model_16.qasm
life_238.qasm
majority_239.qasm
max46_240.qasm
miller_11.qasm
mini-alu_167.qasm
mini_alu_305.qasm
misex1_241.qasm
mlp4_245.qasm
mod10_171.qasm
mod10_176.qasm
mod5adder_127.qasm
mod5d1_63.qasm
mod5d2_64.qasm
mod5mils_65.qasm
mod8-10_177.qasm
mod8-10_178.qasm
multiply_n13.qasm
one-two-three-v0_97.qasm
one-two-three-v0_98.qasm
one-two-three-v1_99.qasm
one-two-three-v2_100.qasm
one-two-three-v3_101.qasm
plus63mod4096_163.qasm
plus63mod8192_164.qasm
pm1_249.qasm
qft_10.qasm
qft_16.qasm
radd_250.qasm
rd32-v0_66.qasm
rd32-v1_68.qasm
rd32_270.qasm
rd53_130.qasm
rd53_131.qasm
rd53_133.qasm
rd53_135.qasm
rd53_138.qasm
rd53_251.qasm
rd53_311.qasm
rd73_140.qasm
rd73_252.qasm
rd84_142.qasm
rd84_253.qasm
root_255.qasm
sao2_257.qasm
sf_274.qasm
sf_276.qasm
sqn_258.qasm
sqrt8_260.qasm
squar5_261.qasm
square_root_7.qasm
sym10_262.qasm
sym6_145.qasm
sym6_316.qasm
sym9_146.qasm
sym9_148.qasm
sym9_193.qasm
sys6-v0_111.qasm
system9.qasm
test.qasm
test1.qasm
test2.qasm
test3.qasm
test4.qasm
urf1_149.qasm
urf1_278.qasm
urf2_152.qasm
urf2_277.qasm
urf3_155.qasm
urf3_279.qasm
urf4_187.qasm
urf5_158.qasm
urf5_280.qasm
urf6_160.qasm
wim_266.qasm
xor5_254.qasm
z4_268.qasm
'''.strip().splitlines()
EXCLUDE = frozenset(['test.qasm','test1.qasm','test2.qasm','test3.qasm','test4.qasm','alu-v0_27_example.qasm'])
FINAL_TEST = frozenset(['wim_266.qasm','squar5_261.qasm'])

# Only previous *CaQR source QASM* split entries are carried over. The other
# ~110 filenames have NO frozen split assignment; leave them "unassigned".
TRAIN = frozenset('''0410184_169.qasm 4mod7-v0_94.qasm alu-v0_27.qasm alu-v2_30.qasm mini-alu_167.qasm mini_alu_305.qasm bv_n10.qasm C17_204.qasm cc_n10.qasm cm152a_212.qasm cm42a_207.qasm cm82a_208.qasm cnt3-5_179.qasm cnt3-5_180.qasm con1_216.qasm decod24-v1_41.qasm graycode6_47.qasm ham7_104.qasm hwb4_49.qasm hwb5_53.qasm majority_239.qasm mod10_176.qasm mod5adder_127.qasm mod5d1_63.qasm mod8-10_178.qasm multiply_n13.qasm one-two-three-v2_100.qasm qft_10.qasm qft_16.qasm rd32_270.qasm rd53_311.qasm rd73_140.qasm rd84_142.qasm sf_276.qasm sym6_316.qasm sym9_146.qasm sys6-v0_111.qasm system9.qasm'''.split())
VALIDATION = frozenset('''ex1_226.qasm 4gt10-v1_81.qasm 4gt11_84.qasm 4gt12-v0_86.qasm 4gt13_92.qasm 4gt4-v0_80.qasm 4gt5_75.qasm ising_model_10.qasm ising_model_16.qasm 4mod5-v0_18.qasm 4mod5-v0_20.qasm mod5mils_65.qasm'''.split())

# Frozen checkpoint filenames; exact matches are development overlaps.
MODEL_TRAIN_FILES = frozenset('''alu-v0_27.qasm cm152a_212.qasm cm42a_207.qasm cnt3-5_179.qasm cnt3-5_180.qasm mini_alu_305.qasm multiply_n13.qasm rd53_311.qasm rd73_140.qasm rd84_142.qasm sym6_316.qasm sym9_146.qasm sys6-v0_111.qasm system9.qasm'''.split())

assert len(NAMES)==168 and len(set(NAMES))==168, len(NAMES)
assert len(EXCLUDE)==6 and all(x in NAMES for x in EXCLUDE)
assert len(set(NAMES)-EXCLUDE)==162
assert not (TRAIN & VALIDATION or TRAIN & FINAL_TEST or VALIDATION & FINAL_TEST)
assert TRAIN|VALIDATION|FINAL_TEST <= set(NAMES)


def family(filename):
    """Coarse exploration grouping; only exact previous split labels are authoritative."""
    s=filename.lower().removesuffix('.qasm')
    if s.startswith('4gt'): return 'four_gt'
    if s.startswith(('4mod5','mod5mils')): return 'modular_arithmetic'
    if s.startswith(('alu','mini_alu','mini-alu')): return 'alu'
    if s.startswith('cm'): return 'cm'
    if s.startswith('rd'): return 'rd'
    if s.startswith(('sym','9symml')): return 'symmetry'
    if s.startswith(('system','sys6')): return 'system'
    if s.startswith('ising_model'): return 'ising_model'
    if s.startswith('qft'): return 'qft'
    if s.startswith('cnt3-5'): return 'cnt3_5'
    if s.startswith(('multiply','plus63mod')): return 'multiplication_related'
    if s.startswith('hwb'): return 'hwb'
    import re
    return re.sub(r'[_-]\d+$','',s)


def roster():
    items=[]
    for name in sorted(set(NAMES)-EXCLUDE):
        split = ('final_test_reserved' if name in FINAL_TEST else
                 'known_train' if name in TRAIN else
                 'known_validation' if name in VALIDATION else 'unassigned')
        items.append({'circuit':name,'family_heuristic':family(name),
                      'previous_split':split,
                      'training_filename_overlap':name in MODEL_TRAIN_FILES,
                      'reserved_final_test':name in FINAL_TEST})
    assert len(items)==162
    return items
