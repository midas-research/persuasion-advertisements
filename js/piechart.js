Chart.defaults.global.defaultFontFamily = "'Roboto', 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif";
Chart.defaults.global.legend.position = 'bottom';
Chart.defaults.global.legend.labels.usePointStyle = true;
Chart.defaults.global.legend.labels.boxWidth = 15;
Chart.defaults.global.tooltips.backgroundColor = '#000';

var colors = [
  ['rgba(185, 70, 70, 1)'],['rgba(185, 99, 70, 1)'],
  ['rgba(185, 128, 70, 1)'],['rgba(185, 156, 70, 1)'],
  ['rgba(185, 185, 70, 1)'],['rgba(156, 185, 70, 1)'], 
  ['rgba(70, 185, 70, 1)'], ['rgba(70, 185, 99, 1)'], 
  ['rgba(70, 185, 128, 1)'], ['rgba(70, 185, 156, 1)'], 
  ['rgba(69, 184, 172, 1)'], ['rgba(70, 185, 185, 1)'], 
  ['rgba(70, 156, 185, 1)'], ['rgba(70, 128, 185, 1)'], 
  ['rgba(70, 99, 185, 1)'], ['rgba(70, 70, 185, 1)'], 
  ['rgba(99, 70, 185, 1)'], ['rgba(128, 70, 185, 1)'], 
  ['rgba(156, 70, 185, 1)'], ['rgba(185, 70, 185, 1)'], 
  ['rgba(185, 70, 156, 1)'], ['rgba(185, 70, 99, 1)']
];

var pieColors = [
  colors[0],colors[1],
  colors[2],colors[3],
  colors[4],colors[5],
  colors[6],colors[7],
  colors[8],colors[9],
  colors[10],colors[11],
  colors[12],colors[13],
  colors[14],colors[15],
  colors[16],colors[17],
  colors[18],colors[19],
  colors[20], colors[21]
];

var my_labels = ["Concreteness","Eager","Fashionable","Creative","Active","Emotion","Cheerful",
                  "Reciprocity","Feminine","Trustworthiness","Unclear","Amazed","Social Identity",
                  "Social Impact","Authority","Anchoring and Comparison","Scarcity","Reverse Psychology",
                  "Foot in the Door","Customer Reviews","Anthropomorphism","Guarantees"];

var my_vals = [1007, 540, 443, 402, 259, 238, 223, 186, 173, 157, 148, 141, 126, 103, 65, 48, 64, 15, 
              18, 28, 37, 45];

// BEGIN PIE CHART

new Chart(document.getElementById("myChart"), {
  type: 'pie',
  data: {
    labels: my_labels,
    datasets: [{
      data: my_vals,
      borderWidth: 2,
      hoverBorderWidth: 10,
      backgroundColor: pieColors,
      hoverBackgroundColor: pieColors,
      hoverBorderColor: pieColors,
      borderColor: pieColors
    }]
  },
  options: {
    legend: {
      position : "right",
      labels: {
        padding: 20
      }
    }
  }
});