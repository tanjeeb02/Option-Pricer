document.addEventListener('DOMContentLoaded', function () {
    let pricingMethodSelect = document.getElementById('pricingMethod');
    let inputGroups = document.querySelectorAll('.input-group');

    let fieldsForMethods = {
        'european_option': ['spotPrice', 'interestRate', 'repoRate', 'maturity', 'strikePrice', 'volatility', 'optionType', 'optionValue', 'calculate'],
        'american_option': ['spotPrice', 'interestRate', 'maturity', 'strikePrice', 'volatility', 'steps', 'optionType', 'optionValue', 'calculate'],
        'asian_geometric_option': ['spotPrice', 'interestRate', 'maturity', 'strikePrice', 'volatility', 'observations', 'optionType', 'isMonte', 'optionValue', 'calculate'],
        'asian_arithmetic_option': ['spotPrice', 'interestRate', 'maturity', 'strikePrice', 'volatility', 'observations', 'paths', 'optionType', 'controlVariate', 'optionValue', 'confidenceInterval', 'calculate'],
        'geometric_basket_option': ['spotPrice', 'interestRate', 'maturity', 'strikePrice', 'spotPriceTwo', 'volatility', 'volatilityTwo', 'correlation', 'optionType', 'isMonte', 'optionValue', 'calculate'],
        'arithmetic_basket_option': ['spotPrice', 'interestRate', 'maturity', 'strikePrice', 'spotPriceTwo', 'volatility', 'volatilityTwo', 'correlation', 'paths', 'optionType', 'controlVariate', 'optionValue', 'confidenceInterval', 'calculate'],
        'kiko_put': ['spotPrice', 'interestRate', 'maturity', 'strikePrice', 'volatility', 'lowerBarrier', 'upperBarrier', 'observations', 'cashRebate', 'optionValue', 'deltaValue', 'calculate']
    };

    pricingMethodSelect.addEventListener('change', function () {
        inputGroups.forEach(function (group) {
            let input = group.querySelector('input');
            let select = group.querySelector('select');

            if (input) {
                if (input.type === 'number' || input.type === 'text') {
                    if (input.id === 'optionValue' || input.id === 'confidenceInterval' || input.id === 'deltaValue') {
                        input.value = '';
                    } else {
                        input.value = input.defaultValue;
                    }
                }
            }

            if (select) {
                select.selectedIndex = 0;
            }

            group.style.display = 'none';
        });

        fieldsForMethods[pricingMethodSelect.value].forEach(function (fieldId) {
            let fieldGroup = document.getElementById(fieldId).closest('.input-group');
            if (fieldGroup) {
                fieldGroup.style.display = 'flex';
            }
        });
    });

    pricingMethodSelect.dispatchEvent(new Event('change'));
});

