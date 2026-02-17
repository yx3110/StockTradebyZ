#!/usr/bin/env python3
"""
Debug V3.81 evaluation_result construction
"""

import sys
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_level4_integrated_system import V380Level4IntegratedSystem

def debug_v381_evaluation():
    print("🔧 Debug V3.81 evaluation_result construction")

    # Initialize V3.81 system
    v381_system = V380Level4IntegratedSystem()

    # Test with a few stocks
    test_codes = ['002211', '835640', '000004']
    test_date = '2025-09-23'

    print(f"🧪 Testing with: {test_codes}")
    print(f"📅 Date: {test_date}")

    # Get V3.81 predictions
    predictions = v381_system.predict_scores_with_quality(test_codes, test_date)

    print(f"\n🔍 Raw predictions type: {type(predictions)}")
    print(f"🔍 Raw predictions keys: {list(predictions.keys()) if isinstance(predictions, dict) else 'Not a dict'}")

    if isinstance(predictions, dict):
        for code, prediction_data in predictions.items():
            print(f"\n📊 {code}:")
            print(f"  prediction_data type: {type(prediction_data)}")
            if isinstance(prediction_data, dict):
                print(f"  keys: {list(prediction_data.keys())}")
                print(f"  overall_score: {prediction_data.get('overall_score', 'MISSING')}")
                print(f"  recommendation: {prediction_data.get('recommendation', 'MISSING')}")
            else:
                print(f"  raw value: {prediction_data}")

    # Now simulate the evaluation_result construction
    print(f"\n🔄 Simulating evaluation_result construction...")

    evaluation_result = {
        'error': False,
        'stocks': []
    }

    for code in test_codes:
        prediction_data = predictions.get(code, {})

        if isinstance(prediction_data, dict):
            overall_score = prediction_data.get('overall_score', 50.0)
            quality_score = prediction_data.get('quality_score', 0.5)
            confidence_score = prediction_data.get('confidence_score', 0.8)
            recommendation = prediction_data.get('recommendation', '观望')
        else:
            overall_score = prediction_data if isinstance(prediction_data, (int, float)) else 50.0
            quality_score = 0.5
            confidence_score = 0.8
            recommendation = '观望'

        stock_result = {
            'code': code,
            'final_score': overall_score / 100.0,
            'overall_score': overall_score,  # Keep original for debugging
            'quality_score': quality_score,
            'confidence_score': confidence_score,
            'recommendation': recommendation
        }
        evaluation_result['stocks'].append(stock_result)

        print(f"  ✅ Added {code}: overall={overall_score}, rec={recommendation}")

    print(f"\n📋 Final evaluation_result:")
    print(f"  error: {evaluation_result['error']}")
    print(f"  stocks count: {len(evaluation_result['stocks'])}")

    v381_stocks = evaluation_result.get('stocks', [])
    print(f"  v381_stocks count: {len(v381_stocks)}")

    if v381_stocks:
        print("  ✅ evaluation_result has stocks - should NOT fallback")
        for stock in v381_stocks:
            print(f"    {stock['code']}: {stock['recommendation']}")
    else:
        print("  ❌ evaluation_result has NO stocks - WILL fallback to traditional analysis")

if __name__ == "__main__":
    debug_v381_evaluation()