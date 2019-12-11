
import java.awt.Color;
import java.awt.Component;
import java.awt.Font;
import javax.swing.UIManager;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Timer;
import java.util.TimerTask;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.mail.MessagingException;
import javax.swing.JOptionPane;
import javax.swing.plaf.FontUIResource;

public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException{
        start form = new start();
        form.setVisible(true);
        readForFall();
        
        //set in start    start.delay = 0; // delay for 0 sec. 
        //set in start   start.period = 3000; // repeat every 3 sec. 
        Timer timer = new Timer(); 
        timer.scheduleAtFixedRate(new TimerTask() 
        { 
        public void run(){ 
            try {
                readForFall();
            } catch (IOException ex) {
                Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            }
        } 
        }, start.delay, start.period); 
    }
    public static void copyFromWifu() throws IOException {
        
    }
  
    public static void readForFall()throws IOException{
        // Switch to the second filename line when running on the desktop
         String filename = "/home/pi/host.txt";
        //String filename = "C://testfile.txt";
        File f = new File(filename);
        BufferedReader br = new BufferedReader(new FileReader(f));
        String line = br.readLine();
        br.close();
        
        //if the file currently contains a 1 (fall), overwrite the filecontents so that falls only get logged once
        if (line.equals("1")){
            start.period = 300000;
            start.userFell = true;
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write("0");
            bw.close();
        }
        else {
            start.period = 3000;
            start.userFell = false;
        }
        
        /*  This tells the GUI what to do when a fall is detected:
        *   1. Log the fall date/time in the local fall data page
        *   2. Begin the mailHandler function, which alerts emergency contacts
        *   3. Interrupt the GUI with a fall alert popup
        */
        if (start.userFell){ 
            start.okay = false;
            String currDate = new java.text.SimpleDateFormat("MM-dd-yyyy").format(new java.util.Date()); //get and format current date
            String currTime = java.time.LocalTime.now().toString(); //get current time
            currTime = currTime.substring(0,8); //trim bc we don't need milliseconds? I don't think?
            java.util.Vector<Object> row = new java.util.Vector<>(); 
            row.add(currDate);
            row.add(currTime);
            start.model.addRow(row); 
            mailHandler(start.contacts, start.count, start.userFullName, currTime, currDate, start.okay);
            Component frame = null;
            UIManager.put("OptionPane.minimumSize",new java.awt.Dimension(800, 440));
            UIManager.put("OptionPane.messageFont", new FontUIResource(new Font("Tahoma", Font.BOLD, 30))); 
            UIManager.put("OptionPane.okButtonText", "I am OK");
            UIManager.put("OptionPane.buttonFont", new FontUIResource(new Font("tahoma",Font.PLAIN,30))); 
            int result=JOptionPane.showConfirmDialog(frame, "<html><font color=#871912>A FALL HAS BEEN DETECTED <br> WIFU will now begin notifying contacts</font></html>", "Fall Alert", JOptionPane.WARNING_MESSAGE, JOptionPane.OK_OPTION);  
            if(result==JOptionPane.OK_OPTION){
                start.okay = true;
                mailHandler(start.contacts, start.count, start.userFullName, currTime, currDate, start.okay);
            }
        }
    }
    
    private static void mailHandler(String[][] phones, int count, String user, String time, String date, boolean okay){
        for(int i = 0; i < count; i++){
            try{
                JavaMailUtil.sendMail(phones[i][4], user, time, date, okay);
            }catch(MessagingException ex){
                Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
}
