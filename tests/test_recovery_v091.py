from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from PIL import Image

from scripts import run_multisource_poc_v09 as runner
from scripts.recovery_support import backup_file, reusable_topup, publish_bioscan
from scripts.fetch_poc_metadata import harvest, convert_occurrence, family_key, INAT_DATASET

ROOT = Path(__file__).resolve().parents[1]


def save_selection(root, minimum=1):
    root.mkdir(parents=True, exist_ok=True)
    selection = root / "diptera_added_families_topup_selection.csv"
    report = root / "diptera_added_families_topup_selection_report.json"
    rows = [{"processid": f"NEW-{i}", "family": family, "order": "Diptera",
             "archive_group": "train", "archive_member": f"train/NEW-{i}.jpg", "source_split": "train"}
            for i, family in enumerate(["Muscidae", "Tachinidae"])]
    pd.DataFrame(rows).to_csv(selection,index=False)
    report.write_text(json.dumps({"families":["Muscidae","Tachinidae"],"min_per_family":minimum,
                                 "topup_selection_rows":2,"by_family":{}}))
    return selection, report


def test_reuse_legacy_selection_without_original_huge_metadata(tmp_path):
    selection, report = save_selection(tmp_path)
    before = selection.read_bytes()
    assert reusable_topup(selection, report, ["Muscidae","Tachinidae"], 1)
    assert selection.read_bytes() == before


def test_incompatible_selection_is_not_overwritten(tmp_path):
    selection, report = save_selection(tmp_path)
    before = selection.read_bytes()
    with pytest.raises(SystemExit, match="NOT overwritten"):
        reusable_topup(selection, report, ["Muscidae","Tachinidae"], 1500)
    assert selection.read_bytes() == before


def test_backup_is_verified_idempotent_and_versions_are_kept(tmp_path):
    original=tmp_path/'manifest.csv';original.write_text('first')
    saved=backup_file(original)
    assert backup_file(original)==saved and saved.read_text()=='first'
    original.write_text('second')
    other=backup_file(original)
    assert other!=saved and saved.read_text()=='first' and other.read_text()=='second'


def test_topup_end_to_end_reuses_saved_selection_and_preserves_old_ids(tmp_path, monkeypatch):
    layout=runner.paths(tmp_path, tmp_path/'raw/bioscan',tmp_path/'manifests/bioscan_raw.parquet')
    root=layout['bioscan'];selection,report=save_selection(root)
    images=root/'diptera_30k_images';images.mkdir()
    for i in range(2): Image.new('RGB',(20,20)).save(images/f'NEW-{i}.jpg')
    oldraw=pd.DataFrame([{'processid':'OLD','family':'Phoridae','order':'Diptera','source_split':'test','local_path':'old.jpg'}])
    oldraw.to_csv(root/'diptera_30k_selection.csv',index=False)
    oldraw.to_csv(root/'diptera_30k_downloaded.csv',index=False)
    layout['bioscan_manifest'].parent.mkdir(parents=True)
    pd.DataFrame([{'source_record_id':'ALREADY-NORMALIZED','family':'Sciaridae','source_split':'test_unseen'}]).to_parquet(layout['bioscan_manifest'],index=False)
    monkeypatch.setattr(runner,'find_bioscan_metadata',lambda *_:pytest.fail('Huge CSV was rescanned'))
    runner.topup_bioscan_families(layout,{'bioscan_family_topup':{'families':['Muscidae','Tachinidae'],'min_images_per_family':1}},False)
    result=pd.read_parquet(layout['bioscan_manifest'])
    assert {'OLD','ALREADY-NORMALIZED','NEW-0','NEW-1'}==set(result.source_record_id)
    assert result.set_index('source_record_id').loc['NEW-0','source_split']=='train'
    assert list(layout['bioscan_manifest'].parent.glob('recovery_backups/*.bak'))
    runner.topup_bioscan_families(layout,{'bioscan_family_topup':{'families':['Muscidae','Tachinidae'],'min_images_per_family':1}},False)
    assert len(pd.read_parquet(layout['bioscan_manifest']))==4


def test_failed_download_does_not_publish_bioscan(tmp_path,monkeypatch):
    layout=runner.paths(tmp_path,tmp_path/'bioscan',tmp_path/'manifest.parquet')
    root=layout['bioscan'];selection,report=save_selection(root)
    pd.DataFrame([{'processid':'OLD'}]).to_csv(root/'diptera_30k_selection.csv',index=False)
    pd.DataFrame([{'processid':'OLD'}]).to_csv(root/'diptera_30k_downloaded.csv',index=False)
    pd.DataFrame([{'source_record_id':'OLD'}]).to_parquet(layout['bioscan_manifest'])
    original=layout['bioscan_manifest'].read_bytes()
    def fail(command,dry_run=False):
        if 'scripts/download_bioscan_subset.py' in command: raise subprocess.CalledProcessError(1,command)
        pytest.fail('Unexpected stage after download failed')
    monkeypatch.setattr(runner,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.topup_bioscan_families(layout,{'bioscan_family_topup':{'families':['Muscidae','Tachinidae'],'min_images_per_family':1}},False)
    assert layout['bioscan_manifest'].read_bytes()==original


def occurrence(source='gbif',license='CC-BY',key=1):
    return {'key':key,'order':'Diptera','family':'Muscidae','genus':'Musca','species':'Musca domestica',
            'basisOfRecord':'PRESERVED_SPECIMEN' if source=='gbif' else 'HUMAN_OBSERVATION',
            'datasetKey':'museum' if source=='gbif' else INAT_DATASET,
            'occurrenceID':f'https://www.inaturalist.org/observations/{key}',
            'media':[{'type':'StillImage','identifier':f'https://inaturalist-open-data.s3.amazonaws.com/photos/{key}/original.jpg','license':license,'creator':'Author'}]}


def test_poc_sources_keep_provenance_and_filter_media_licenses():
    row=convert_occurrence(occurrence('inat'),'inat','Muscidae')
    assert row['source']=='iNaturalist' and row['source_record_id']=='1'
    assert row['label_quality']=='C' and not row['is_preserved_specimen']
    assert row['image_url'].endswith('/medium.jpg') and row['attribution']=='Author'
    row=convert_occurrence(occurrence(),'gbif','Muscidae')
    assert row['source']=='GBIF' and row['source_record_id']=='1' and row['is_preserved_specimen']
    assert convert_occurrence(occurrence(license='CC-BY-NC'),'gbif','Muscidae') is None
    assert convert_occurrence(occurrence(),'inat','Muscidae') is None


class FakeAPI:
    def __init__(self): self.calls=[]
    def get(self,endpoint,params=None):
        self.calls.append((endpoint,params))
        if endpoint.startswith('/dataset/'): return {'title':'iNaturalist Research-grade Observations'}
        if endpoint=='/species/match': return {'usageKey':5564,'matchType':'EXACT','rank':'FAMILY','canonicalName':'Muscidae','order':'Diptera'}
        return {'results':[occurrence('inat',key=i) for i in range(1,4)],'endOfRecords':True}


def test_bounded_poc_resumes_without_network_and_enforces_cap(tmp_path):
    client=FakeAPI();out=tmp_path/'poc.parquet'
    report=harvest('inat',out,['Muscidae'],2,client)
    assert report['rows']==2 and len(pd.read_parquet(out))==2
    no_network=Mock();no_network.get.side_effect=AssertionError('Unexpected network')
    assert harvest('inat',out,['Muscidae'],2,no_network)['rows']==2
    assert client.calls[-1][1]['datasetKey']==INAT_DATASET


def test_poc_checkpoint_mismatch_preserves_existing(tmp_path):
    out=tmp_path/'poc.parquet';harvest('inat',out,['Muscidae'],2,FakeAPI())
    before=out.read_bytes()
    with pytest.raises(RuntimeError,match='settings differ'): harvest('inat',out,['Muscidae'],3,FakeAPI())
    assert before==out.read_bytes()


def test_ingest_preflight_returns_all_missing_paths_without_cascade(tmp_path):
    config=runner.source_config(runner.paths(tmp_path))
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    p=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_corpus.py'),'--config',str(path),'--stage','ingest'],capture_output=True,text=True)
    assert p.returncode!=0
    assert 'STOP:' in p.stderr and 'inaturalist/poc_api.parquet' in p.stderr and 'dissco/diptera_full.jsonl' in p.stderr
    assert 'Traceback' not in p.stderr


def test_notebook_compiles_and_no_unchecked_shell_stages():
    nb=json.loads((ROOT/'notebooks/TaxaLens_v0.9_Multisource_PoC_Colab.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert '!python' not in code and '!pip' not in code
    assert "stage('checkpoint')" in code and "stage('download','--source','inat'" in code
    assert "stage('quality')" in code
    assert "if result:" in code
    for c in nb['cells']:
        if c['cell_type']=='code': compile(''.join(c['source']),'<cell>','exec')


def test_interrupted_manifest_write_preserves_completed_file(tmp_path):
    from diptera_id.corpus.io import ManifestWriter
    out=tmp_path/'existing.parquet'
    with ManifestWriter(out) as writer: writer.write([{'source_record_id':'OLD'}])
    before=out.read_bytes()
    with pytest.raises(RuntimeError):
        with ManifestWriter(out) as writer:
            writer.write([{'source_record_id':'NEW'}])
            raise RuntimeError('simulated interruption')
    assert out.read_bytes()==before


def test_current_dissco_full_response_uses_asset_not_doi():
    from scripts.ingest_dissco import specimen_rows
    obj={'data':{'id':'https://doi.org/10.3535/SPECIMEN','attributes':{
        'digitalSpecimen':{'dwc:order':'Diptera','dwc:family':'Muscidae','dwc:basisOfRecord':'PreservedSpecimen'},
        'digitalMedia':[{'digitalMediaObject':{'@id':'https://doi.org/10.3535/MEDIA',
            'dcterms:identifier':'https://doi.org/10.3535/MEDIA','dcterms:type':'StillImage',
            'ac:accessURI':'https://example.org/actual.jpg','dcterms:rights':'https://creativecommons.org/publicdomain/zero/1.0/legalcode'},
            'annotations':[{'identifier':'https://doi.org/not-an-image'}]}]}}}
    result=specimen_rows(obj,False)
    assert len(result)==1 and result[0]['image_url']=='https://example.org/actual.jpg'
    assert result[0]['image_license']=='CC0'
    obj['data']['attributes']['digitalSpecimen']['dwc:basisOfRecord']='FossilSpecimen'
    assert specimen_rows(obj,False)==[]


def test_dissco_resume_walks_past_already_saved_page(tmp_path,monkeypatch):
    from scripts import download_dissco as d
    out=tmp_path/'dissco.jsonl'
    out.write_text(json.dumps({'data':{'id':'https://doi.org/10.3535/OLD'}})+'\n')
    calls=[]
    def fake_get(session,url,params,retries):
        calls.append(params)
        if params:
            assert params['order']=='Diptera' and params['livingOrPreserved']=='Preserved' and params['hasMedia']=='true'
            pid='OLD' if params['pageNumber']==1 else 'NEW'
            return {'data':[{'id':'https://doi.org/10.3535/'+pid}]}
        return {'data':{'id':'https://doi.org/10.3535/NEW'}}
    monkeypatch.setattr(d,'get_json',fake_get)
    monkeypatch.setattr(sys,'argv',['download_dissco.py','--out',str(out),'--resume','--max-records','2','--page-size','1','--sleep','0'])
    d.main()
    assert len(out.read_text().splitlines())==2
    assert any(p and p['pageNumber']==2 for p in calls)


def test_four_source_ingest_assemble_plan_with_bounded_manifests(tmp_path):
    from diptera_id.corpus.io import ManifestWriter
    layout=runner.paths(tmp_path,tmp_path/'raw/bioscan',tmp_path/'manifests/bioscan_raw.parquet')
    for source in ['inat','gbif']:
        with ManifestWriter(layout[source]/'poc_api.parquet') as writer:
            writer.write([convert_occurrence(occurrence(source,key=10 if source=='inat' else 20),source,'Muscidae')])
    with ManifestWriter(layout['bioscan_manifest']) as writer:
        writer.write([{'source':'BIOSCAN-5M','source_record_id':'BIO-1','image_url':'https://example.org/bio1.jpg','image_license':'CC-BY','order':'Diptera','family':'Tachinidae','source_split':'train'},
                      {'source':'BIOSCAN-5M','source_record_id':'BIO-2','image_url':'https://example.org/bio2.jpg','image_license':'CC-BY','order':'Diptera','family':'Muscidae','source_split':'test_unseen'}])
    original=layout['bioscan_manifest'].read_bytes()
    layout['dissco'].mkdir()
    (layout['dissco']/'diptera_full.jsonl').write_text(json.dumps({'digitalSpecimenId':'D-1','order':'Diptera','family':'Muscidae',
        'ods:hasMedia':[{'digitalMediaId':'DM-1','mediaType':'StillImage','accessURI':'https://example.org/dissco1.jpg','license':'CC0'}]})+'\n')
    for stage in ['ingest','assemble','plan']:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_multisource_poc_v09.py'),'--stage',stage,'--data-root',str(tmp_path),
                        '--bioscan-root',str(layout['bioscan']),'--bioscan-manifest',str(layout['bioscan_manifest'])],cwd=ROOT,check=True)
    plan=pd.read_parquet(layout['plan'])
    assert set(plan.source)=={'iNaturalist','GBIF','DiSSCo','BIOSCAN-5M'}
    assert layout['bioscan_manifest'].read_bytes()==original
