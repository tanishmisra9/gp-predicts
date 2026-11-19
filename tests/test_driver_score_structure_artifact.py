"""Independent indexed-projection/old-projection/coordinate/lap/publication audit."""
import gzip,json,math
from collections import defaultdict,Counter
from datetime import datetime,timedelta,timezone
from email.utils import parsedate_to_datetime
from f1.data import ROOT


def entries(node):
    if node is None:return []
    if isinstance(node,list):return list(enumerate(node))
    return [(int(k),v) for k,v in node.items()]


def test_artifact():
    result=json.loads((ROOT/'results/e231_structure_summary.json').read_text())
    previous=json.loads((ROOT/'results/e230_native_schema.json').read_text())
    old={(r['season'],r['round']):r for r in previous['sources'] if r['feed']=='DriverScore'}
    expected_dates={5:'2019-05-12',12:'2019-08-04',14:'2019-09-08'}
    assert len(result['sources'])==3
    good=0;total=0
    for r in result['sources']:
        assert r['available'] and r['raw_sha256']==old[(r['season'],r['round'])]['raw_sha256']
        assert r['requests']==[dict(url=r['url'],status=200)] and r['status']==200
        assert r['url'].endswith('_Race/DriverScore.jsonStream')
        data=json.loads(gzip.decompress((ROOT/f'results/e231_2019_{r["round"]}_indexed.json.gz').read_bytes()))
        prior=json.loads(gzip.decompress((ROOT/f'results/e230_2019_{r["round"]}_DriverScore_numeric.json.gz').read_bytes()))
        assert len(data['records'])==len(prior)==r['records'] and r['malformed']==0
        observed=[]
        def old_paths(node,path):
            leaves=[]
            if isinstance(node,list):
                for value in node:leaves.extend(old_paths(value,path+['[]']))
            elif isinstance(node,dict):
                for k,value in node.items():leaves.extend(old_paths(value,path+[k]))
            elif node is not None:leaves.append(dict(path=path,value=node))
            return leaves
        for record,earlier in zip(data['records'],prior):
            assert record['seconds']==earlier['seconds']
            assert old_paths(record['scores'],['Scores'])==earlier['numeric']
            for car,rows in record['scores'].items():
                for row,components in entries(rows):
                    for component,value in entries(components):
                        if value is not None:
                            observed.append(dict(record=record['record'],seconds=record['seconds'],car=car,row=row,component=component,value=value))
        assert observed==data['observations'];total+=len(observed)
        histories=defaultdict(list)
        for o in observed:histories[o['car'],o['row'],o['component']].append(o)
        terminal={key:updates[-1]['value'] for key,updates in histories.items()}
        known={(l['car'],l['lap']):l['time'] for l in data['laps']}
        axes={(c,row):v for (c,row,comp),v in terminal.items() if comp==0}
        values=defaultdict(list)
        for (c,row,comp),v in terminal.items():
            if comp!=0:values[comp].append(v)
        variable={comp for comp,v in values.items() if min(v)<max(v)}
        covered=defaultdict(set);matches=0;first=[];last=[]
        for (car,row),lap in axes.items():
            if lap!=row+1 or (car,int(lap)) not in known:continue
            matches+=1;present=0
            for component in variable:
                key=(car,row,component)
                if key not in histories:continue
                present+=1;first.append(histories[key][0]['seconds']-known[car,int(lap)])
                last.append(histories[key][-1]['seconds']-known[car,int(lap)])
            if present>=2:covered[car].add(row)
        count=sum(len(v)>=10 for v in covered.values());fraction=matches/len(axes)
        assert r['axis_rows']==len(axes) and r['axis_matches']==matches and r['axis_match_fraction']==fraction
        assert r['variable_components']==sorted(variable) and r['nonaxis_components']==sorted(values)
        assert r['component_ranges']=={str(c):[min(v),max(v)] for c,v in values.items()}
        assert Counter(first)==Counter(r['first_measurement_delays']) and Counter(last)==Counter(r['last_measurement_delays'])
        assert r['postlap_fraction']==sum(v>=-2 for v in first)/len(first)
        assert r['drivers_with_10_rows_2_components']==count
        revisions=sum(len({o['value'] for o in updates})>1 for updates in histories.values())
        assert revisions==r['revised_coordinate_count'] and len(histories)==r['numeric_coordinates']
        structural=len(axes)>=100 and fraction>=.95 and count>=10 and len(variable)>=2
        limit=datetime.fromisoformat(expected_dates[r['round']]).replace(tzinfo=timezone.utc)+timedelta(days=4)
        assert limit.isoformat()==r['publication_limit']
        modified=parsedate_to_datetime(r['last_modified']);timely=modified<=limit
        assert timely==r['publication_gate']==False and structural==r['structure_gate']==False
        good+=int(structural and timely and not r['unknown_car_keys'])
        print('2019',r['round'],'matched bytes, old/new projections, axis',matches,'/',len(axes),'publication',timely)
    assert result['gate']==(good>=2)==False and not result['winner_fit'] and not result['channel_names_verified']
    assert total==69413
    print('All69413 numeric values and structural/provenance failure independently verified')


if __name__=='__main__':test_artifact()
