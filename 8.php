<?php
if(isset($_POST['mysubmit']))
{
  $a = $_POST['txt1'];
  $b = $_POST['txt2'];
  $c = $_POST['txt3'];
  $d = $_POST['txt4'];
  $e = $_POST['txt5'];

  $con=mysqli_connect("localhost","root","avinash","storevalue");
 
 if ($con->connect_error) {
            die("Connection failed: " . $con->connect_error);
          }
          
          $sql = "INSERT INTO storevaluetable VALUES ('$a','$b','$c','$d','$e','SUCCESS')";
          
          if ($con->query($sql) === TRUE)
          {
            echo "<script>alert('CLICK OK TO PAYMENT PROCESS');</script>";
          }
          else
          {
            echo "Error: " . $sql . "<br>" . $con->error;
          }
          
          $con->close();
}
?>
<html>
<center>
<style>
html, body {margin: 0; height: 100%; overflow: hidden}
</style>
<body scroll="no" style="overflow: hidden">
<style>
body{background-image: url(webimage2.jpg); background-position : center; background-repeat : no-repeat; background-size : cover;}

</style>
<body>
<br><br><br><br><br><br><br><br><br><br>
<b><font size = "4" family = "Times New Roman">Click for payment process--------> </b><a href = "paymentinfo.html">payment</a></font>
<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
<pre>
<b>Under   section   107   of   the   Copyright   Act   copyrighted   works  can  be  used  or  borrowed  without  the  creators  permission  for   fair use   purposes   such   as   education   or   commentary.
                 PRIVACY POLICY | TERMS OF USE
                 1986-2022  FAST TRAVEL
</b>
</html>