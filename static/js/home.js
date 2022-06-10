// // 로그아웃
// $(function () {
//     $('#write').click(() => {
       
//                 location.href = "./write.html";
            
        
//     });
// });

// // 서버추가 버튼
// $(function () {
//     $('#sever_add').click(() => {
//         let token = localStorage.getItem('access_token')
//         // console.log(token);
//         var server_name = $('#server_name').val(); // 서버명
//         // console.log(server_name)

//         $.ajax({
//             headers: {
//                 "authorization": 'bearer ' + token,
//             },
//             data: {
//                 'srv_name': server_name
//             },
//             url: 'https://49.50.174.207:5000/server?srv_name=' + server_name,
//             type: "POST",
//             dataType: "json",
//             success: (data) => {
//                 // console.log(data)
//                 let result = data.result;

//                 if (result == "1") {
//                     alert("'" + server_name + "' 서버가 추가되었습니다.")
//                     // console.log("추가된 서버명", server_name)
//                     $('#exampleModal').modal('hide'); // 모달창 닫기
//                     window.location.reload(); // 새로고침
//                 }
//                 else {
//                     alert("오류가 발생하였습니다. 다시 실행해주시길 바랍니다.")
//                 }
//             },
//             error: function (error) {
//                 console.log(error)
//             },
//             complete: function (data) {

//             }
//         });
//     });
// });
