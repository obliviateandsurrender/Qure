const ctx = document.getElementById("myChart").getContext("2d");
let delayed;
//Gradient Fill
let gradient = ctx.createLinearGradient(0, 0, 0, 400);
gradient.addColorStop(0, "rgba(58, 123, 213, 1)");
gradient.addColorStop(0, "rgba(0, 210, 255, 0.3)");
const labels = [
  "2012",
  "2013",
  "2014",
  "2015",
  "2016",
  "2017",
  "2018",
  "2019",
  "2020",
];

fetch('https://file.io/Qhz2JhERCg1M')
  .then(res => res.blob()) // Gets the response and returns it as a blob
  .then(blob => {
    // Here's where you get access to the blob
    // And you can use it for whatever you want
    // Like calling ref().put(blob)

    // Here, I use it to make an image appear on the page
    let objectURL = URL.createObjectURL(blob);
    // FileReader Object
    var reader = new FileReader();
    reader.readAsText(res.blob())

    reader.onload = function (event) {
        // Read file data
        var csvdata = event.target.result;
        // Split by line break to gets rows Array
        var rowData = csvdata.split("\n");
        var data = rowData[1];
        // Loop on the row Array (change row=0 if you also want to read 1st row)
        // for (var row = 1; row < 2; row++) {
        //   // Split by comma (,) to get column Array
        //   data.datasets.data = rowData[row].split(",");
        // }
        console.log(data);
      };
    return data
});

// function readCSVFile() {
//   // FileReader Object
//   var reader = new FileReader();
//   // Read file as string
//   reader.readAsText("data1.csv");
//   // Load event
//   reader.onload = function (event) {
//     // Read file data
//     var csvdata = event.target.result;
//     // Split by line break to gets rows Array
//     var rowData = csvdata.split("\n");
//     var data = rowData[1];
//     // Loop on the row Array (change row=0 if you also want to read 1st row)
//     // for (var row = 1; row < 2; row++) {
//     //   // Split by comma (,) to get column Array
//     //   data.datasets.data = rowData[row].split(",");
//     // }
//     return data;
//   };
// }
const data = {
  labels,
  datasets: [
    {
      data: [],
      //   [211, 326, 165, 350, 420, 370, 500, 375, 415],
      label: "Heart Rate",
      fill: true,
      backgroundColor: gradient,
      borderColor: "#fff",
      pointBackgroundColor: "#fff",
    },
  ],
};
const config = {
  type: "line",
  data: data,
  options: {
    radius: 5,
    hitRadius: 30,
    hoverRadius: 12,
    responsive: true,
    animation: {
      onComplete: () => {
        delayed = true;
      },
      delay: (context) => {
        let delay = 0;
        if (context.type === "data" && context.mode === "default" && !delayed) {
          delay = context.dataIndex * 300 + context.datasetIndex * 100;
        }
        return delay;
      },
    },
    scales: {
      y: {
        ticks: {
          callback: function (value) {
            return "$" + value + "m";
          },
        },
      },
    },
  },
};
const myChart = new Chart(ctx, config);