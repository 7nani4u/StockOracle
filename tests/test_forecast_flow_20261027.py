import json
import subprocess

from api.index import HTML


def test_forecast_follows_decision_evidence_scenarios_entry_execution_risk():
    tab = HTML.split('<div id="tab-forecast"', 1)[1].split('<!-- 뉴스 탭 -->', 1)[0]
    ids = ['prediction-overview-section', 'prediction-ai-context-section',
           'prediction-scenarios-section', 'buy-price-section', 'box-execution-section', 'risk-grid']
    positions = [tab.index(f'id="{key}"') for key in ids]
    assert positions == sorted(positions)
    assert '.forecast-guide:empty{display:none}' in HTML


def test_execution_renderer_is_separate_and_clears_old_plan():
    source = 'function renderPredictionSections' + HTML.split('function renderPredictionSections', 1)[1].split('function renderTechnicalSignals', 1)[0]
    data = {'prediction_outlook': {'decision': {'key': 'watch'}},
            'box_range': {'tactics': {'regime': {'key': 'range_support', 'tradeable': True,
                         'label': '지지 확인'}, 'summary': '시험 실행',
                         'plan': {'status': 'wait', 'tradeable': False, 'entry': {'low': 90, 'high': 91},
                                  'geometry': {'first_target_rr': 1.5, 'breakeven_rate': .57,
                                               'measured_rate': None, 'ok': None},
                                  'entry_checks': [{'label': '확정 일봉 재시험 지지', 'ok': False}]}}},
            'buy_price': {'box_stage_display': True}, 'risk_scenarios': None}
    script = r'''
const vm = require('vm');
const elements = {};
const context = {
 document: {getElementById(id) {return elements[id] ||= {innerHTML:'',textContent:'',hidden:false};}},
 _isFiniteNumber(v) {return v != null && Number.isFinite(Number(v));},
 _escPrediction(v) {return String(v ?? '');}, _predictionTone() {return '#aaa';},
 fmt(v) {return String(v);}, fmtRange(a,b) {return `${a}~${b}`;}
};
vm.createContext(context); vm.runInContext(SOURCE, context);
const data = DATA;
context.renderForecast(data, false);
const plan = elements['box-execution-section']?.innerHTML || '';
const entries = elements['buy-price-section'].innerHTML;
data.box_range = null; context.renderForecast(data, false);
console.log(JSON.stringify({plan, entries, cleared:elements['box-execution-section'] || {innerHTML:'missing',hidden:false}}));
'''.replace('SOURCE', json.dumps(source)).replace('DATA', json.dumps(data))
    result = subprocess.run(['node', '-'], input=script, text=True, encoding='utf-8',
                            capture_output=True, check=True, timeout=20)
    rendered = json.loads(result.stdout)
    assert '박스 실행 플랜' in rendered['plan']
    assert '대기' in rendered['plan']
    assert '비교 자료 부족' in rendered['plan']
    assert '감사 0%' not in rendered['plan']
    assert '확정 일봉 재시험 지지' in rendered['plan']
    assert '박스 실행 플랜' not in rendered['entries']
    assert rendered['cleared']['innerHTML'] == ''
    assert rendered['cleared']['hidden'] is True

