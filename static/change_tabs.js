document.addEventListener('DOMContentLoaded', function () {
    let optionTab = document.getElementById('optionTab');
    let volatilityTab = document.getElementById('volatilityTab');
    let optionCalculator = document.getElementById('optionCalculator');
    let impliedVolatility = document.getElementById('impliedVolatility');
    let calculateButton = document.getElementById('calculate');
    let tabs = [optionTab, volatilityTab];

    function removeActiveClass() {
        tabs.forEach(tab => {
            tab.classList.remove('active-tab');
        });
    }

    function showOptionCalculator() {
        optionCalculator.style.display = 'block';
        impliedVolatility.style.display = 'none';

        removeActiveClass()
        optionTab.classList.add('active-tab');
    }

    function showImpliedVolatility() {
        let inputGroups = impliedVolatility.querySelectorAll('.input-group');
        inputGroups.forEach(group => group.style.display = 'flex');

        optionCalculator.style.display = 'none';
        impliedVolatility.style.display = 'block';

        removeActiveClass()
        volatilityTab.classList.add('active-tab');
    }


    optionTab.addEventListener('click', function (event) {
        event.preventDefault();
        showOptionCalculator();
    });

    volatilityTab.addEventListener('click', function (event) {
        event.preventDefault();
        showImpliedVolatility();
    });

    calculateButton.addEventListener('click', function () {
        if (optionCalculator.style.display === 'block') {
        } else if (impliedVolatility.style.display === 'block') {
        }
    });

    showOptionCalculator();
});
