// (function()
// {
// 	var canvas = document.querySelector( "#canvas" );
// 	var context = canvas.getContext( "2d" );
// 	canvas.width = 224;
// 	canvas.height = 224;

// 	var Mouse = { x: 0, y: 0 };
// 	var lastMouse = { x: 0, y: 0 };
// 	context.fillStyle="white";
// 	context.fillRect(0,0,canvas.width,canvas.height);
// 	context.color = "black";
// 	context.lineWidth = 4;
//     context.lineJoin = context.lineCap = 'round';
	
// 	debug();

// 	canvas.addEventListener( "mousemove", function( e )
// 	{
// 		lastMouse.x = Mouse.x;
// 		lastMouse.y = Mouse.y;

// 		Mouse.x = e.pageX - this.offsetLeft;
// 		Mouse.y = e.pageY - this.offsetTop;

// 	}, false );

// 	canvas.addEventListener( "mousedown", function( e )
// 	{
// 		canvas.addEventListener( "mousemove", onPaint, false );

// 	}, false );

// 	canvas.addEventListener( "mouseup", function()
// 	{
// 		canvas.removeEventListener( "mousemove", onPaint, false );

// 	}, false );

// 	var onPaint = function()
// 	{	
// 		context.lineWidth = context.lineWidth;
// 		context.lineJoin = "round";
// 		context.lineCap = "round";
// 		context.strokeStyle = context.color;
	
// 		context.beginPath();
// 		context.moveTo( lastMouse.x, lastMouse.y );
// 		context.lineTo( Mouse.x, Mouse.y );
// 		context.closePath();
// 		context.stroke();
// 	};

// 	function debug()
// 	{
// 		/* CLEAR BUTTON */
// 		var clearButton = $( "#clearButton" );
		
// 		clearButton.on( "click", function()
// 		{
			
// 				context.clearRect( 0, 0, 224, 224 );
// 				context.fillStyle="white";
// 				context.fillRect(0,0,canvas.width,canvas.height);
			
// 		});

// 		/* COLOR SELECTOR */

// 		$( "#colors" ).change(function()
// 		{
// 			var color = $( "#colors" ).val();
// 			context.color = color;
// 		});
		
// 		/* LINE WIDTH */
		
// 		$( "#lineWidth" ).change(function()
// 		{
// 			context.lineWidth = $( this ).val();
// 		});

// 	}
// }());

// $("#submitCanvas").click(function(){
// 	var canvasObj = document.getElementById("canvas");
// 	var img = canvasObj.toDataURL();
// 	var filename = document.getElementById("fname").value;
//     var data = JSON.stringify(canvas_data);
     
//     $.post("/", { save_fname: filename, save_cdata: data });
// });