<?php
if(isset($_POST['mysubmit']))
{
  $a=$_POST['txt1'];
  $b=$_POST['txt2'];
  $c=$_POST['txt3'];

 $con=mysqli_connect("localhost","root","avinash","storevalue");
 
 $que="select * from storevaluetable natural join busdetails where tname = '$a' and location = '$b' and destination = '$c'";
 
 $result=mysqli_query($con,$que);
}
?>
<html>
<center>
<style>
html,body{margin: 0; height: 100%; overflow: hidden}
</style>
<body scroll="no" style="overflow: hidden">
<style>
body{background-image: url(webimage2.jpg); background-position : center; background-repeat : no-repeat; background-size : cover;}

table{
  border-collapse: collapse;
  width: 20%;
}

th,td{
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}
frame{overflow:hidden;}
</style>
<body>
<font size = "4"><b><u>RECEIPT</u></b></font>
<br><br><br>
<table border = "2"  width = "5">
<tr><th>NAME</th><th>DATE</th><th>NUMBER OF SEATS</th><th>BUS NUMBER</th><th>SOURCE</th><th>DESTINATION</th><th>PAYMENT</th>

    <?php while($row=mysqli_fetch_Assoc($result))
        {
            ?>
      <tr>
      <tr><td><b><?php print_r($row['tname']); ?></b></td>
      <td><b><?php print_r($row['tdate']); ?></b></td>
		<td><b><?php print_r($row['tnumberofseats']); ?></b></td>
		<td><b><?php print_r($row['busnumber']); ?></b></td>
       <td><b><?php print_r($row['location']); ?></b></td>
       <td><b><?php print_r($row['destination']); ?></b></td>
       <td><b><?php print_r($row['tpayment']); ?></td></b></tr>


                <?php
                }
                ?>
</table>
<br><br><br><br>
<style>
.button {
  background-color: #4CAF50; /* Green */
  border: none;
  color: white;
  padding: 8px 6px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 10px;
}
</style>
<button onClick="window.print()" class = "button">print</button>
<pre>
</b>
</html>
