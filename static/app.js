
init();

function init(){
    let text2predict=""  
}

function button_predict(){
  let text_2_predict=document.getElementById("predictclick").value
  let text_length=text_2_predict.length
  if (text_length <= 30)  {
    window.confirm("Please enter text with more than 30 characters")
    var t1 = document.getElementById("predictclick")
    t1.value = ""
  }
  else {
    //console.log(text2predict)
    //console.log(text2predict.length)
    //url = `http://127.0.0.1:5000/mbti_predict/${text2predict}`
    //url = `https://mb-type-predictor.herokuapp.com/mbti_predict/${text2predict}`
    url = "/mbti_predict/" + text_2_predict
    //url = url_for('mbti_predict', text2predict=text_2_predict)
    //window.confirm(url)
    //url = "{{url_for('mbti_predict')}}"
    //url = url + '/' + text_2_predict
    //console.log(url)
    d3.json(url).then((mbti) => {
        console.log(mbti)
        window.confirm(`"Personality Type Based on Text is,${mbti}"`)
        var p1 = document.getElementById("person-type")
        p1.innerHTML=mbti
        var t1 = document.getElementById("predictclick")
        t1.value = ""
    })
  }
}


/* function countryOptionChanged(countrySelected){
    // Get the year range
    let selYear = document.getElementById("selYear");
    let yearRangeVal  = selYear.options[selYear.selectedIndex].value;
    restyle(countrySelected, yearRangeVal);

}

function countryOptionChanged2(countrySelected2){
    create_allEmissions_line_chart(countrySelected2); 
    create_allConsumption_line_chart(countrySelected2);   
}

function yearsOptionChanged(yearRangeVal){
    // Get the year range
    let selCountry = document.getElementById("selCountry");
    let countrySelected  = selCountry.options[selCountry.selectedIndex].value;
    restyle(countrySelected, yearRangeVal);
}

 */
