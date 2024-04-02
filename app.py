from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pricer import Pricer
import traceback
import logging

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('interface.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    method = data.get('pricingMethod')
    calculation_type = data.get('calculationType')

    if calculation_type == 'optionPrice':
        pricer = Pricer(float(data['spotPrice']), float(data['interestRate']), float(data['repoRate']),
                        float(data['maturity']),
                        float(data['strikePrice']))
        method_mapping = {
            'european_option': {
                'func': pricer.european_option,
                'params': ['volatility', 'optionType']
            },
            'american_option': {
                'func': pricer.american_option,
                'params': ['volatility', 'steps', 'optionType']
            },
            'asian_geometric_option': {
                'func': pricer.asian_geometric_option,
                'params': ['volatility', 'observations', 'optionType', 'isMonte']
            },
            'asian_arithmetic_option': {
                'func': pricer.asian_arithmetic_option,
                'params': ['volatility', 'observations', 'paths', 'optionType', 'controlVariate']
            },
            'geometric_basket_option': {
                'func': pricer.geometric_basket_option,
                'params': ['spotPriceTwo', 'volatility', 'volatilityTwo', 'correlation', 'optionType',
                           'isMonte']
            },
            'arithmetic_basket_option': {
                'func': pricer.arithmetic_basket_option,
                'params': ['spotPriceTwo', 'volatility', 'volatilityTwo', 'correlation', 'paths', 'optionType',
                           'controlVariate']
            },
            'kiko_put': {
                'func': pricer.kiko_put,
                'delta_func': pricer.kiko_put_delta,
                'params': ['spotPrice', 'volatility', 'lowerBarrier', 'upperBarrier', 'observations', 'cashRebate']
            }
        }
        if method not in method_mapping:
            return jsonify({'error': 'Invalid pricing method'}), 400

        func_info = method_mapping[method]
        func = func_info['func']
        required_params = func_info['params']

        params = {}
        for param in required_params:
            if param in data:
                if param in ['optionType']:
                    params[param] = str(data[param])
                elif param in ['steps', 'observations', 'paths']:
                    params[param] = int(data[param])
                elif param in ['isMonte', 'controlVariate']:
                    params[param] = data[param] == "True"
                else:
                    params[param] = float(data[param])

        try:
            # for param_name, param_value in params.items():
            #     print(f"{param_name}: {param_value}, Type: {type(param_value)}")
            optionPrice = func(*params.values())
            response_data = {'optionPrice': optionPrice}
            if method in ['asian_arithmetic_option', 'arithmetic_basket_option']:
                response_data['optionPrice'] = optionPrice[0]
                response_data['confInterval'] = optionPrice[1]
            elif method == 'kiko_put':
                delta_func = func_info['delta_func']
                if delta_func:
                    delta = delta_func(*params.values())
                    response_data['deltaKiko'] = delta
            print(response_data)
            return jsonify(response_data)

        except Exception as e:
            print(f"Error type: {type(e).__name__}, Error message: {e}")
            traceback_details = traceback.format_exc()
            print(traceback_details)
            logging.exception("Exception occurred during calculation:")
            return jsonify({'error': str(e)}), 500

    elif calculation_type == 'impliedVolatility':
        try:
            spot = float(data.get('ivSpotPrice'))
            interest = float(data.get('ivInterestRate'))
            repo = float(data.get('ivRepoRate'))
            maturity = float(data.get('ivMaturity'))
            strike = float(data.get('ivStrikePrice'))
            premium = float(data.get('optionPremium'))
            option_type = data.get('ivOptionType')

            pricer = Pricer(spot, interest, repo, maturity, strike)

            volatility = pricer.iv(premium, option_type)
            return jsonify({'impliedVolatility': volatility})

        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': 'Invalid calculation type'}), 400


if __name__ == '__main__':
    app.run(debug=True)
