document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('calculate').addEventListener('click', calculatePrice);
});

function calculatePrice() {
    let activeFormId = document.getElementById('impliedVolatility').style.display === 'block' ? 'volatilityForm' : 'optionForm';
    let formData = new FormData(document.getElementById(activeFormId));

    let object = {};
    formData.forEach((value, key) => object[key] = value);

    if (activeFormId === 'optionForm') {
        object['calculationType'] = 'optionPrice';
        object['pricingMethod'] = document.getElementById('pricingMethod').value;
    } else if (activeFormId === 'volatilityForm') {
        object['calculationType'] = 'impliedVolatility';
    }

    let json = JSON.stringify(object);

    fetch('/calculate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: json,
    })
        .then(response => response.json())
        .then(data => {
            if (activeFormId === 'optionForm') {
                document.getElementById('optionValue').value = data.optionPrice;
                if ('confInterval' in data) {
                    document.getElementById('confidenceInterval').value = `[${data.confInterval.join(', ')}]`;
                }
                else if ('deltaKiko' in data) {
                    document.getElementById('deltaValue').value = data.deltaKiko;
                }
            } else if (activeFormId === 'volatilityForm') {
                document.getElementById('ivResult').value = data.impliedVolatility;
            }
        })
        .catch(error => console.error('Error:', error));
}
