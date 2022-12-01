
$('#birth').datetimepicker({
    format: 'MM/DD/YYYY'
});

$('#state').select2({
    theme: "bootstrap"
});

/* validate */

// validation when select change
$("#state").change(function () {
    $(this).valid();
})

// validation when inputfile change
$("#uploadImg").on("change", function () {
    $(this).parent('form').validate();
})


jQuery.validator.addMethod("lettersonly", function(value, element) {
  return this.optional(element) || /^[a-zA-Z0-9_]+$/i.test(value);
}, "Vui lòng nhập chữ cái hoặc số"); 


$("#formValidation").validate({
    validClass: "success",
    rules: {
        uploadsize: {
            required: true
        },
        userstoresize: {
            required: true
        },
    },
    highlight: function (element) {
        $(element).closest('.form-group').removeClass('has-success').addClass('has-error');
    },
    success: function (element) {
        $(element).closest('.form-group').removeClass('has-error').addClass('has-success');
    },
});

$("#formValidation").validate({
    validClass: "success",
    rules: {
        confirmpassword: {
            equalTo: "#password"
        },
        user: {
            lettersonly: true
        },
        birth: {
            date: true
        },
        uploadImg: {
            required: true,
        },
        email: {
            email: true
        },
        addEmail: {
            email: true
        }
    },
    highlight: function (element) {
        $(element).closest('.form-group').removeClass('has-success').addClass('has-error');
    },
    success: function (element) {
        $(element).closest('.form-group').removeClass('has-error').addClass('has-success');
    },
});

$("#formEditValidation").validate({
    validClass: "success",
    rules: {
        confirmpassword: {
            equalTo: "#password"
        },
        birth: {
            date: true
        },
        uploadImg: {
            required: true,
        },
        email: {
            email: true
        },
        addEmail: {
            email: true
        }
    },
    highlight: function (element) {
        $(element).closest('.form-group').removeClass('has-success').addClass('has-error');
    },
    success: function (element) {
        $(element).closest('.form-group').removeClass('has-error').addClass('has-success');
    },
});



// form kiểm tra mật khẩu validate
$("#formPassWordPros").validate({
    validClass: "success",
    rules: {
        password: {
            required: true
        },
        newpassword: {
            required: true
        },
        confirmpassword: {
            equalTo: "#newpassword"
        },
    },
    highlight: function (element) {
        $(element).closest('.form-group').removeClass('has-success').addClass('has-error');
    },
    success: function (element) {
        $(element).closest('.form-group').removeClass('has-error').addClass('has-success');
    },
});


// input chọn giờ
$('#checkin').datetimepicker({
    format: 'h:mm A',
}).on('changeDate', function(e){
    $(this).datepicker('hide');
});

$('#checkout').datetimepicker({
    format: 'h:mm A',
}).on('changeDate', function(e){
    $(this).datepicker('hide');
});
$('#editCheckin').datetimepicker({
    format: 'h:mm A',
}).on('changeDate', function(e){
    $(this).datepicker('hide');
});

$('#editCheckout').datetimepicker({
    format: 'h:mm A',
}).on('changeDate', function(e){
    $(this).datepicker('hide');
});

 
